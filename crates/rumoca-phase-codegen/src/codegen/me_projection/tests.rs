//! Render-time refusals of the ME projection view.

use rumoca_eval_solve::TargetIsolationProgram;

use super::block::isolation_kind;

/// An isolation the evaluator answers but no scalar program reproduces
/// refuses its block before any byte, and never interns an isolator.
///
/// `TargetIsolationProgram::Unrepresentable` arises only when an assignment
/// shape's materialization does not fit its row (a register range overflow or
/// a tensor-affine lane outside its source range), which no shape derived from
/// a lowered model produces today; this pins the refusal the renderer applies
/// if one ever does.
#[test]
fn an_unrepresentable_row_isolation_refuses_its_block() {
    let mut interned = 0;
    let mut intern = || {
        interned += 1;
        3
    };
    let error = isolation_kind(7, &TargetIsolationProgram::Unrepresentable, &mut intern)
        .expect_err("an unrepresentable isolation must refuse the block");
    assert_eq!(
        isolation_kind(
            7,
            &TargetIsolationProgram::Isolator(Vec::new()),
            &mut intern
        )
        .ok(),
        Some((2, 3)),
        "a materialized isolator is interned exactly once"
    );
    let message = error.to_string();
    assert!(
        message.contains("unsupported-feature:algebraic_projection")
            && message.contains("projection block 7")
            && message.contains("row isolation no scalar program reproduces"),
        "{message}"
    );
    for (program, kind) in [
        (TargetIsolationProgram::Unavailable, 0),
        (TargetIsolationProgram::OutputValue, 1),
    ] {
        assert_eq!(
            isolation_kind(7, &program, &mut intern).ok(),
            Some((kind, 0))
        );
    }
    assert_eq!(
        interned, 1,
        "only the materialized isolator is interned; the refused isolation interns nothing"
    );
}

/// Lane seeds are checked in their own units: a `LoadSeed` index is a
/// position among `seed_len * lanes`, a tensor load's seed range is in seeds.
#[test]
fn lane_seed_loads_are_checked_in_their_own_units() {
    use rumoca_ir_solve::{LinearOp, TensorInputKind};

    use super::block::check_seed_loads;

    let (seed_len, lanes) = (4, 3);
    let load = |index| vec![LinearOp::LoadSeed { dst: 0, index }];
    let tensor = |seed_start| {
        vec![LinearOp::TensorLoad {
            dst_start: 0,
            input: TensorInputKind::Y,
            input_start: 0,
            count: 2,
            seed_start: Some(seed_start),
            lanes: lanes + 1,
        }]
    };
    assert!(check_seed_loads(0, &load(seed_len * lanes - 1), seed_len, lanes).is_ok());
    assert!(check_seed_loads(0, &load(seed_len * lanes), seed_len, lanes).is_err());
    assert!(check_seed_loads(0, &tensor(seed_len - 2), seed_len, lanes).is_ok());
    // Within `seed_len * lanes` positions but past the last seed.
    assert!(check_seed_loads(0, &tensor(seed_len - 1), seed_len, lanes).is_err());
}

/// A dispatched program stays out of line past the inlining bound only, so a
/// small program may be inlined into its case and a large one keeps its own
/// function.
#[test]
fn a_dispatched_program_stays_out_of_line_only_past_the_inlining_bound() {
    let program = |ops: usize| {
        minijinja::context! {
            temporary_count => 1,
            ops => vec![minijinja::context! { kind => "Move", dst => 0, src => 0 }; ops],
        }
    };
    let mut environment = crate::codegen::create_environment();
    environment
        .add_template(
            "inlining.c",
            r#"{%- from "fmi-me-projection.jinja" import program_function -%}
{{ program_function("rmc_row_p0", small, "ModelInstance* m, double* o", false, true) }}
{{ program_function("rmc_row_p1", large, "ModelInstance* m, double* o", false, true) }}"#,
        )
        .expect("the inlining template parses");
    let rendered = environment
        .get_template("inlining.c")
        .expect("the inlining template")
        .render(minijinja::context! {
            small => program(super::INLINE_PROGRAM_MAX_OPS),
            large => program(super::INLINE_PROGRAM_MAX_OPS + 1),
        })
        .expect("render the two programs");
    assert!(rendered.contains("static int rmc_row_p0("), "{rendered}");
    assert!(
        rendered.contains("static RMC_NOINLINE int rmc_row_p1("),
        "{rendered}"
    );
}

/// Isolator groups slice within a bound on their code: slots that each
/// repeat a long shared prefix stay one shared group, while slots that each
/// read their own element of a tensor operation, as a ladder's do, slice.
mod isolation_slices {
    use rumoca_ir_solve::{BinaryOp, LinearOp, TensorInputKind};

    use super::super::table::isolation_slices;

    /// A chain of `prefix` products every one of `slots` stores reads, each
    /// slot adding its own parameter.
    fn shared_prefix(prefix: u32, slots: u32) -> Vec<LinearOp> {
        let mut program = vec![LinearOp::LoadY { dst: 0, index: 0 }];
        for k in 1..=prefix {
            program.push(LinearOp::Binary {
                dst: k,
                op: BinaryOp::Mul,
                lhs: k - 1,
                rhs: 0,
            });
        }
        for slot in 0..slots {
            let base = 100 + 2 * slot;
            program.extend([
                LinearOp::LoadP {
                    dst: base,
                    index: slot as usize,
                },
                LinearOp::Binary {
                    dst: base + 1,
                    op: BinaryOp::Add,
                    lhs: prefix,
                    rhs: base,
                },
                LinearOp::StoreOutput { src: base + 1 },
            ]);
        }
        program
    }

    /// A load of `slots` elements, then for each slot a chain of `chain`
    /// products of its own element, which it stores: the shape of a ladder
    /// cell's isolations.
    fn per_slot_chains(slots: usize, chain: u32) -> Vec<LinearOp> {
        let mut program = vec![LinearOp::TensorLoad {
            dst_start: 0,
            input: TensorInputKind::Y,
            input_start: 0,
            count: slots,
            seed_start: None,
            lanes: 1,
        }];
        for slot in 0..slots as u32 {
            let base = 1000 + slot * (chain + 1);
            program.push(LinearOp::Move {
                dst: base,
                src: slot,
            });
            for k in 1..=chain {
                program.push(LinearOp::Binary {
                    dst: base + k,
                    op: BinaryOp::Mul,
                    lhs: base + k - 1,
                    rhs: slot,
                });
            }
            program.push(LinearOp::StoreOutput { src: base + chain });
        }
        program
    }

    #[test]
    fn slots_sharing_a_long_prefix_stay_one_group() {
        let program = shared_prefix(40, 8);
        let slices = isolation_slices(&program, 8);
        assert_eq!(slices.len(), 8);
        assert!(slices.iter().all(Option::is_none), "{slices:?}");
        // Two slots repeat the prefix within the bound and slice.
        assert!(
            isolation_slices(&shared_prefix(40, 2), 2)
                .iter()
                .all(Option::is_some)
        );
    }

    #[test]
    fn slots_reading_their_own_elements_slice() {
        let program = per_slot_chains(50, 18);
        let slices = isolation_slices(&program, 50);
        assert!(slices.iter().all(Option::is_some));
        let code: usize = slices.iter().flatten().map(Vec::len).sum();
        assert!(code <= 2 * program.len(), "{code} for {}", program.len());
        assert!(slices.iter().flatten().all(|slice| slice.len() == 21));
    }
}

/// A device build runs a long sequence of whole programs as out-of-line runs
/// of at most the device bound in operations, called in order; a program past
/// the bound forms a run of its own.
#[test]
fn a_device_program_sequence_runs_in_bounded_out_of_line_runs() {
    let program = |ops: usize| {
        minijinja::context! {
            temporary_count => 1,
            ops => vec![minijinja::context! { kind => "Move", dst => 0, src => 0 }; ops],
        }
    };
    let half = super::DEVICE_CHUNK_MAX_OPS / 2;
    let programs = vec![
        program(half),
        program(half),
        program(1),
        program(super::DEVICE_CHUNK_MAX_OPS + 1),
        program(1),
    ];
    let mut environment = crate::codegen::create_environment();
    environment
        .add_template(
            "runs.cu",
            r#"{%- from "fmi-me-projection.jinja" import device_runs, program_block -%}
{%- call(program) device_runs("rmc_seq", programs, "ModelInstance* m, double* o", "m, o", "(void)m; (void)o;") %}
{{- program_block(program, "o") }}
{%- endcall %}"#,
        )
        .expect("the runs template parses");
    let rendered = environment
        .get_template("runs.cu")
        .expect("the runs template")
        .render(minijinja::context! { programs => programs })
        .expect("render the program sequence");
    for run in 0..4 {
        assert!(
            rendered.contains(&format!(
                "static __device__ RMC_NOINLINE int rmc_seq_k{run}(ModelInstance* m, double* o) {{"
            )),
            "{rendered}"
        );
        assert!(
            rendered.contains(&format!("    if (rmc_seq_k{run}(m, o) != 0) return -1;")),
            "{rendered}"
        );
    }
    assert!(!rendered.contains("rmc_seq_k4"), "{rendered}");
    assert!(
        rendered.contains("static __device__ int rmc_seq(ModelInstance* m, double* o) {"),
        "{rendered}"
    );
    let run = |k: usize| {
        let start = rendered
            .find(&format!("rmc_seq_k{k}(ModelInstance"))
            .expect("the run");
        let end = rendered[start..].find("    return 0;\n}").expect("its end") + start;
        rendered[start..end].matches("double r[").count()
    };
    assert_eq!(
        [run(0), run(1), run(2), run(3)],
        [2, 1, 1, 1],
        "the two halves fill one run, and the program past the bound stands alone between its neighbours"
    );
}

/// A device dispatcher of more programs than one group selects an
/// out-of-line group of consecutive programs; the host dispatcher is the one
/// switch over every program.
#[test]
fn a_device_dispatcher_selects_out_of_line_program_groups() {
    let group = super::DEVICE_CHUNK_MAX_OPS / super::INLINE_PROGRAM_MAX_OPS;
    let mut environment = crate::codegen::create_environment();
    environment
        .add_template(
            "dispatch.c",
            r#"{%- from "fmi-me-projection.jinja" import dispatcher -%}
{{ dispatcher("static __device__" if device else "RMC_API", "row", count, "size_t program, ModelInstance* m, double* out", "m, out", device) }}"#,
        )
        .expect("the dispatcher template parses");
    let render = |device: bool| {
        environment
            .get_template("dispatch.c")
            .expect("the dispatcher template")
            .render(minijinja::context! { count => group + 1, device => device })
            .expect("render the dispatcher")
    };
    let host = render(false);
    assert!(
        host.starts_with("RMC_API int rmc_row_call(size_t program, ModelInstance* m, double* out) {\n    switch (program) {\n    case 0: return rmc_row_p0(m, out);"),
        "{host}"
    );
    assert!(!host.contains("_g0"), "{host}");
    let device = render(true);
    assert!(
        device.contains("static __device__ RMC_NOINLINE int rmc_row_call(size_t program, ModelInstance* m, double* out) {")
            && device.contains(&format!("    switch (program / {group}) {{"))
            && device.contains("    case 1: return rmc_row_call_g1(program, m, out);")
            && device.contains(&format!("    case {group}: return rmc_row_p{group}(m, out);"))
            && !device.contains("rmc_row_call_g2"),
        "{device}"
    );
}

/// A device switch over sparse case values (the rows with their own function
/// beside shared row shapes) groups each value by its range of one group's
/// width and selects only the groups that hold a value.
#[test]
fn a_device_switch_groups_sparse_cases_by_range() {
    let group = super::DEVICE_CHUNK_MAX_OPS / super::INLINE_PROGRAM_MAX_OPS;
    let ids = vec![0, 5, group + 3, group + 4, 3 * group];
    let mut environment = crate::codegen::create_environment();
    environment
        .add_template(
            "sparse.cu",
            r#"{%- from "fmi-me-projection.jinja" import grouped_switch -%}
{{ grouped_switch("static __device__", "rmc_row_own_call", "program", ids, "rmc_row_p", "size_t program, ModelInstance* m, double* out", "m, out", "program, m, out", "rmc_bad_program(m, program)")|trim }}"#,
        )
        .expect("the sparse switch template parses");
    let rendered = environment
        .get_template("sparse.cu")
        .expect("the sparse switch template")
        .render(minijinja::context! { ids => ids })
        .expect("render the sparse switch");
    assert!(
        rendered.starts_with(
            "static __device__ RMC_NOINLINE int rmc_row_own_call_g0(size_t program, ModelInstance* m, double* out) {"
        ),
        "{rendered}"
    );
    for (group_index, id) in [
        (0, 0),
        (0, 5),
        (1, group + 3),
        (1, group + 4),
        (3, 3 * group),
    ] {
        assert!(
            rendered.contains(&format!("    case {id}: return rmc_row_p{id}(m, out);")),
            "{rendered}"
        );
        assert!(
            rendered.contains(&format!(
                "    case {group_index}: return rmc_row_own_call_g{group_index}(program, m, out);"
            )),
            "{rendered}"
        );
    }
    assert!(!rendered.contains("rmc_row_own_call_g2"), "{rendered}");
    assert!(
        rendered.contains(&format!("    switch (program / {group}) {{")),
        "{rendered}"
    );
}
