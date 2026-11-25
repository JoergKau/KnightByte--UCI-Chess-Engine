//! Negamax search: the node function.
//!
//! The node function runs these stages in order: stop checks, draw detection,
//! mate-distance pruning, the check extension (bounded by a per-path budget),
//! tablebase probing, the first quiescence drop, the transposition-table probe,
//! the static evaluation, evaluation-based pruning (reverse futility,
//! razoring), null move, the singular-extension verification, internal
//! iterative reduction, a second quiescence drop for nodes the depth
//! adjustments consumed, ProbCut, and finally the ordered move loop with its
//! per-move pruning, extensions, late move reductions and the
//! principal-variation search of each move.
//!
//! Each stage that is more than a few lines lives in its own small file and
//! this function only applies the verdict:
//!
//! | stage | file |
//! | :--- | :--- |
//! | table probe, cutoffs, narrowing | `tt_probe.rs` |
//! | static evaluation, correction, hindsight | `node_eval.rs` |
//! | reverse futility, razoring | `eval_prune.rs` (rules in `pruning.rs`) |
//! | null move | `null_move.rs` |
//! | singular verification | `singular.rs` |
//! | ProbCut | `probcut.rs` |
//! | SEE, late-move, history, futility, bad-noisy pruning; negative extension; cutoff rewards | `move_rules.rs` |
//! | reduction schedule, re-search depth | `lmr.rs` |
//! | the scout and re-search of one move | `search_move.rs` |
//!
//! The `cut_node` flag is threaded through the recursion: a child's prediction
//! depends on how its parent was searched. The `excluded_move` parameter makes
//! the singular-extension verification search the alternatives only (that
//! verification never extends, a known inert mechanism: quirk Q1).
//!
//! Ordering state is borrowed only while the picker is alive: the ordered move
//! list is copied into a small local array first, so recursion never has to
//! hold a borrow of `self`.

use crate::board::movegen;
use crate::board::position::Position;
use crate::search::eval_prune::EvalPruneNode;
use crate::search::history::{CONTINUATION_DISTANCES, Continuations, Previous};
use crate::search::lmr::{LmrInput, lmr_applies, lmr_reduction};
use crate::search::move_rules::{MoveContext, QuietFacts, QuietPrune, quiet_prune};
use crate::search::node_eval::{NodeEval, NodeEvalQuery};
use crate::search::null_move::NullMoveNode;
use crate::search::probcut::ProbCutNode;
use crate::search::pruning::{futility_margin, iir_applies, mate_distance_window};
use crate::search::search_move::ChildSearch;
use crate::search::searcher::{
    DRAW_SCORE, INFINITE, MATE_LIMIT, NodeResult, SearchEvent, Searcher, mated_in,
};
use crate::search::singular::SingularNode;
use crate::search::tt::Bound;
use crate::search::tt_probe::{TtQuery, TtStep};
use crate::types::r#move::Move;
use crate::types::piece::{PAWN, color_of, piece_type};

/// The node at and beyond this ply returns its static evaluation without
/// counting, matching the reference's safety guard.
const MAX_SEARCH_PLY: usize = 100;

impl Searcher {
    /// Returns the futility margin for `depth`, scaled by the lever.
    fn scaled_futility_margin(&self, depth: i32) -> i32 {
        futility_margin(depth) * self.params.futility_scale / 100
    }

    /// Builds the continuation-history context of the node at `ply`: the
    /// destination squares of the moves played 1, 2, 4 and 6 plies earlier.
    pub(crate) fn continuations(&self, ply: usize) -> Continuations {
        let mut cont = [None; 4];
        for (index, &distance) in CONTINUATION_DISTANCES.iter().enumerate() {
            if ply >= distance {
                let mv = self.move_stack[ply - distance];
                if !mv.is_none() {
                    cont[index] = Some(mv.to());
                }
            }
        }
        cont
    }

    /// Searches `position` to `depth` plies and returns its score.
    ///
    /// `previous` identifies the move that produced `position`; it selects the
    /// continuation-history and countermove entries. `ply` is the distance
    /// from the root and indexes the per-ply arrays. `cut_node` predicts a
    /// fail-high node (children of scouting searches). `excluded_move` is the
    /// transposition-table move a singular verification must not search.
    /// `path_extensions` is the extension budget already spent along this
    /// path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn negamax<const STATS: bool>(
        &mut self,
        position: &Position,
        mut alpha: i32,
        mut beta: i32,
        mut depth: i32,
        ply: usize,
        previous: Option<Previous>,
        cut_node: bool,
        excluded_move: Move,
        path_extensions: i32,
        observer: &mut dyn FnMut(SearchEvent),
    ) -> i32 {
        // The ply guard runs before the node is counted: a node beyond the
        // safety depth returns its static evaluation and costs nothing. No
        // accumulator pair exists at this ply, so the full refresh runs.
        if ply >= MAX_SEARCH_PLY {
            return self.evaluator.evaluate(position);
        }
        self.pv.clear(ply);
        self.nodes += 1;
        crate::node_stat!(NegamaxNodes);
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        if self.should_stop() {
            return 0;
        }
        let is_root = ply == 0;
        if !is_root && self.is_draw(position, ply) {
            return DRAW_SCORE;
        }
        self.keys[ply] = position.key();

        // Mate-distance pruning is skipped at the root, which must complete its
        // move loop and report a real best move.
        if !is_root {
            (alpha, beta) = mate_distance_window(alpha, beta, ply);
            if alpha >= beta {
                return alpha;
            }
        }

        // Check extension: a check extends the node itself, within the
        // per-path extension budget.
        let in_check = position.in_check(position.side());
        let mut extensions_used = path_extensions;
        let cap = self.params.max_path_extensions;
        if in_check && extensions_used < cap {
            depth += 1;
            extensions_used += 1;
            if STATS {
                self.stats.check_extensions += 1;
            }
        }

        // Exact tablebase values (`tb.rs`). The root is excluded so it always
        // completes its move loop and reports a real best move.
        let pv_node = beta > alpha + 1;
        if !is_root && let Some(score) = self.tb_node_score(position, pv_node, depth) {
            return score;
        }

        if depth <= 0 {
            return self.quiescence_counted::<STATS>(position, alpha, beta, ply, previous);
        }
        if STATS {
            self.stats.record_main_node(depth);
        }

        // The exact-bound decision compares against the window as it stood
        // before the transposition-table narrowing below.
        let alpha_window = alpha;

        // Transposition-table probe (the stage lives in `tt_probe.rs`). A
        // position that already repeats on the search path is refused table
        // cutoffs.
        let allow_tt_cutoff = !self.params.disable_tt_cutoff_for_repetitions
            || self.repetition_count(position, ply) == 0;
        let tt = match self.probe_table::<STATS>(
            position,
            TtQuery {
                alpha,
                beta,
                depth,
                ply,
                pv_node,
                cut_node,
                is_root,
                allow_cutoff: allow_tt_cutoff,
            },
        ) {
            TtStep::Return(score) => return score,
            TtStep::Continue(node) => node,
        };
        alpha = tt.alpha;
        beta = tt.beta;
        depth = tt.depth;
        let mut tt_move = tt.tt_move;
        let tt_probe = tt.entry;
        let tt_eval = tt.eval;

        // DTZ root ranking: try the 50-move-aware tablebase move first. The
        // picker validates pseudo-legality, the root restriction filters it,
        // and a repeating move evaluates as a draw at its child, so this stays
        // an ordering hint rather than a bypass.
        if is_root && self.params.use_syzygy_dtz_root_ranking && !self.root_tb_move.is_none() {
            tt_move = self.root_tb_move;
        }

        // The static evaluation (the stage lives in `node_eval.rs`): the
        // network's value, corrected, replaced by a consistent stored score,
        // and the hindsight depth pair.
        let NodeEval {
            raw_eval,
            eval,
            depth: hindsight_depth,
        } = self.evaluate_node(
            position,
            NodeEvalQuery {
                depth,
                ply,
                in_check,
                pv_node,
                previous,
                tt_move,
                tt_eval,
                tt_probe,
                allow_tt_cutoff,
            },
        );
        depth = hindsight_depth;

        let improving = self.is_improving(ply, eval);

        // Evaluation-based pruning (the stage lives in `eval_prune.rs`):
        // reverse futility and razoring.
        if !pv_node
            && !in_check
            && excluded_move.is_none()
            && let Some(score) = self.eval_prune::<STATS>(
                position,
                EvalPruneNode {
                    alpha,
                    beta,
                    depth,
                    ply,
                    eval,
                    improving,
                    previous,
                },
            )
        {
            return score;
        }

        // Null move (the stage lives in `null_move.rs`): the side to move
        // passes, and a fail-high of the reduced search ends the node.
        if !is_root
            && !in_check
            && let Some(score) = self.null_move_cutoff::<STATS>(
                position,
                NullMoveNode {
                    beta,
                    depth,
                    ply,
                    cut_node,
                    eval,
                    tt_probe,
                    extensions_used,
                },
                observer,
            )
        {
            return score;
        }

        // Singular extension (the stage lives in `singular.rs`): the number
        // of extra plies the table move earns in the move loop below.
        let singular_extension = if self.params.use_singular_extensions
            && !is_root
            && excluded_move.is_none()
            && !tt_move.is_none()
        {
            let Some(plies) = self.singular_extension::<STATS>(
                position,
                SingularNode {
                    depth,
                    ply,
                    previous,
                    cut_node,
                    tt_move,
                    tt_probe,
                    extensions_used,
                },
                observer,
            ) else {
                return 0;
            };
            plies
        } else {
            0
        };
        if STATS && singular_extension > 0 {
            self.stats.singular_extensions += 1;
        }

        // Internal iterative reduction (the rule lives in
        // `pruning::iir_applies`): a node that expects a full search and has no
        // usable table move searches one ply shallower, so the search finds a
        // move to order by first.
        if !in_check
            && excluded_move.is_none()
            && iir_applies(
                &self.params,
                depth,
                pv_node || is_root || cut_node,
                !tt_move.is_none(),
                tt_probe.map(|entry| i32::from(entry.depth)),
            )
        {
            depth -= 1;
            if STATS {
                self.stats.iir_reductions += 1;
            }
        }

        if depth <= 0 {
            return self.quiescence_counted::<STATS>(position, alpha, beta, ply, previous);
        }

        // ProbCut (off by default, lever `UseProbCut`; the stage lives in
        // `probcut.rs`): a capture that proves a fail-high by a shallow search
        // ends the node early.
        if self.params.use_probcut
            && !pv_node
            && !in_check
            && excluded_move.is_none()
            && let Some(score) = self.probcut::<STATS>(
                position,
                ProbCutNode {
                    beta,
                    depth,
                    ply,
                    cut_node,
                    improving,
                    eval,
                    raw_eval,
                    tt_move,
                    tt_probe,
                    extensions_used,
                },
                observer,
            )
        {
            return score;
        }

        crate::node_stat!(NegamaxMoveLists);
        let moves = movegen::legal(position);
        crate::node_stat!(PseudoMoves, moves.as_slice().len());
        if moves.is_empty() {
            return if in_check {
                mated_in(ply as i32)
            } else {
                DRAW_SCORE
            };
        }

        // `go searchmoves`: at the root, only the listed moves are searched.
        let restricted = if ply == 0 {
            self.root_restriction(moves.as_slice())
        } else {
            Vec::new()
        };
        let moves_slice: &[Move] = if restricted.is_empty() {
            moves.as_slice()
        } else {
            &restricted
        };

        let cont = self.continuations(ply);
        self.picker[ply].reset(
            tt_move,
            &self.killers,
            &self.history,
            ply,
            previous,
            &cont,
            false,
            false,
        );

        let futile = !pv_node
            && !in_check
            && depth <= self.params.futility_max_depth
            && eval + self.scaled_futility_margin(depth) < alpha;

        let mut best_score = -INFINITE;
        let mut best_move = Move::NONE;
        let mut quiet_move_count = 0;
        let mut legal_moves = 0;
        let mut quiets_searched = [Move::NONE; 64];
        let mut quiets_count = 0;

        while let Some(mv) = self.picker[ply].next(position, &self.history, moves_slice) {
            crate::node_stat!(MovesTried);
            // Step 10: the reference's single `info currmove` line of the
            // search — emitted when iteration 1's first root move begins,
            // before the depth-1 line, with `currmovenumber 1` and the node
            // count at that moment (1). No line when the search never
            // reaches iteration 1's move loop (a stop before it).
            if ply == 0 && depth == 1 && !self.currmove_reported {
                self.currmove_reported = true;
                observer(SearchEvent::CurrMove {
                    mv,
                    number: 1,
                    nodes: self.nodes,
                });
            }
            if !excluded_move.is_none() && mv == excluded_move {
                continue;
            }
            let capture = movegen::is_capture(position, mv);
            let quiet_counted = !capture && !mv.is_promotion();
            let moving = position.piece_at(mv.from());

            // Combined history score, computed before the move is made.
            let history_score = self.move_history_score(position, mv, capture, moving, &cont);
            let context = MoveContext {
                depth,
                in_check,
                pv_node,
                capture,
                moving,
                history_score,
                ply,
                previous,
            };

            // Depth-scaled SEE pruning (the rule lives in `move_rules.rs`).
            if self.see_prunes::<STATS>(position, mv, context, legal_moves) {
                continue;
            }

            self.move_stack[ply] = mv;

            // The legal-move counter increments immediately after the
            // legality check and before every pruning block, so pruned moves
            // still occupy a slot in the count late-move reductions index by.
            legal_moves += 1;
            self.move_counts[ply] = legal_moves;
            if quiet_counted {
                quiet_move_count += 1;
            }

            // Root-move node attribution for the time manager's
            // node-fraction term: the whole subtree of this move (including
            // the re-searches) lands in its slot.
            let nodes_before_move = if is_root { self.nodes } else { 0 };

            // Quiet-move pruning (late moves, history, futility; the rules
            // live in `move_rules.rs`). The futility flag is re-tested against
            // the current alpha, which rises as the loop finds better moves.
            if quiet_counted {
                let facts = QuietFacts {
                    depth,
                    in_check,
                    pv_node,
                    improving,
                    legal_moves,
                    quiet_move_count,
                    history_score,
                    pawn_move: piece_type(moving) == PAWN,
                    futile: futile && eval + self.scaled_futility_margin(depth) < alpha,
                };
                match quiet_prune(&self.params, facts) {
                    QuietPrune::Keep => {}
                    QuietPrune::LateMoves => {
                        if STATS {
                            self.stats.lmp_skips += 1;
                        }
                        self.picker[ply].skip_quiets();
                        continue;
                    }
                    QuietPrune::History => {
                        if STATS {
                            self.stats.history_skips += 1;
                        }
                        continue;
                    }
                    QuietPrune::Futility => {
                        if STATS {
                            self.stats.futility_skips += 1;
                        }
                        self.picker[ply].skip_quiets();
                        continue;
                    }
                }
            }

            // Bad-noisy pruning ends the move loop at the first losing capture
            // when even a winning capture could not reach alpha; the margin
            // becomes the node's fail-soft floor.
            if let Some(margin) =
                self.bad_noisy_stop(position, mv, context, eval, alpha, legal_moves)
            {
                if best_score.abs() < MATE_LIMIT && best_score <= margin {
                    best_score = margin;
                }
                break;
            }

            // The move is made, and the child's accumulator pair pushed, only
            // once the pruning blocks above have let it through: they decide
            // from the parent's data alone, so a pruned move costs neither a
            // board copy nor a network update. The pair at ply + 1 is written
            // from the parent's pair by every move that reaches a child, so
            // a skipped push leaves nothing stale behind.
            let next = position.make_move(mv);
            self.tables.tt.prefetch(next.key());
            crate::node_stat!(MovesMade);
            if STATS {
                self.stats.moves_made += 1;
            }
            self.push_accumulators(position, &next, mv, ply);
            let gives_check = next.in_check(next.side());

            let mut extension = if mv == tt_move { singular_extension } else { 0 };

            // Negative extension (`move_rules.rs`): a losing capture or a quiet
            // the history tables distrust is searched one ply shallower.
            extension += self.negative_extension::<STATS>(position, mv, context, gives_check);

            let new_depth = (depth - 1 + extension).max(0);

            // Late move reduction (the schedule lives in `lmr.rs`): later quiet
            // moves are searched shallower, by how much depending on the log
            // table, the history of the move, the node type and a few signals.
            let mut reduction = 0;
            if lmr_applies(
                &self.params,
                depth,
                legal_moves,
                capture || mv.is_promotion(),
                in_check || gives_check,
            ) {
                // Killers are read live; the countermove is the value the
                // picker captured before the loop started (see
                // `PickerState::counter_move`).
                let killer_or_counter = self.killers.contains(ply, mv)
                    || self.picker[ply].counter_move() == mv.without_double_push();
                reduction = lmr_reduction(
                    &self.params,
                    LmrInput {
                        base: self.lmr_base_reduction(depth, legal_moves),
                        mover_pieces_left: next.occupancy(color_of(moving)).count_ones(),
                        improving,
                        history_score,
                        cut_node,
                        pv_node,
                        killer_or_counter,
                        new_depth,
                    },
                );
                if STATS {
                    self.stats.record_lmr(reduction);
                }
            }
            // Both per-ply stacks are published here, per searched move, and
            // not at node entry: a null-move child (or any node reached
            // before the first move of this node is searched) therefore sees
            // the values an earlier visit of this ply left behind. The
            // hindsight pair and the improving flag depend on that.
            self.reduction_stack[ply] = reduction;
            self.evals[ply] = eval;

            let score = self.search_child::<STATS>(
                ChildSearch {
                    child: &next,
                    alpha,
                    beta,
                    new_depth,
                    reduction,
                    ply,
                    previous: Some((moving, mv.from(), mv.to())),
                    pv_node,
                    cut_node,
                    first_move: legal_moves == 1,
                    extensions_used,
                },
                observer,
            );
            if self.stopped {
                return 0;
            }

            if is_root {
                self.nodes_per_move[usize::from(mv.from()) * 64 + usize::from(mv.to())] +=
                    self.nodes - nodes_before_move;
            }

            if score <= alpha && quiet_counted && quiets_count < quiets_searched.len() {
                quiets_searched[quiets_count] = mv;
                quiets_count += 1;
            }

            if score > best_score {
                best_score = score;
                best_move = mv;
                if score > alpha {
                    alpha = score;
                    self.pv.update(ply, mv);
                    if score >= beta {
                        if STATS {
                            self.stats.record_cutoff((legal_moves - 1) as usize);
                        }
                        self.reward_cutoff(
                            position,
                            mv,
                            context,
                            &quiets_searched[..quiets_count],
                            &cont,
                        );
                        return self.finish_node(NodeResult {
                            position,
                            ply,
                            depth,
                            score: best_score,
                            raw_eval,
                            corrected_eval: eval,
                            bound: Bound::Lower,
                            best_move,
                            in_check,
                            previous,
                            excluded: !excluded_move.is_none(),
                        });
                    }
                }
            }
        }

        // No legal move was searched. Besides a genuine mate or stalemate this
        // is the singular-verification case where the excluded table move is
        // the only legal move: the node then reports the mate or draw score
        // (nothing is stored), which is what the extension tiers compare
        // against.
        if legal_moves == 0 {
            return if in_check {
                mated_in(ply as i32)
            } else {
                DRAW_SCORE
            };
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > alpha_window {
            Bound::Exact
        } else {
            Bound::Upper
        };
        let stored_move = if bound == Bound::Upper {
            tt_move
        } else {
            best_move
        };
        self.finish_node(NodeResult {
            position,
            ply,
            depth,
            score: best_score,
            raw_eval,
            corrected_eval: eval,
            bound,
            best_move: stored_move,
            in_check,
            previous,
            excluded: !excluded_move.is_none(),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::types::r#move::{Move, flags};

    #[test]
    fn a_table_move_keeps_its_castle_and_en_passant_markers() {
        // Bug B35 fix (reference bug 147): the stored move comes back exactly
        // as written, so a stored castling or en-passant move compares equal
        // to the generated move it came from.
        let castle = Move::new(4, 6, flags::CASTLE);
        assert_eq!(castle.flag(), flags::CASTLE);

        let en_passant = Move::new(36, 43, flags::EN_PASSANT);
        assert_eq!(en_passant.flag(), flags::EN_PASSANT);
    }

    #[test]
    fn other_table_moves_are_kept_as_stored() {
        for mv in [
            Move::new(12, 28, flags::DOUBLE_PUSH),
            Move::new(52, 60, flags::PROMO_QUEEN),
            Move::new(52, 60, flags::PROMO_KNIGHT),
            Move::new(6, 21, flags::NORMAL),
            Move::NONE,
        ] {
            assert_eq!(Move::from_bits(mv.bits()), mv);
        }
    }
}
