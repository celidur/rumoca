//! Level-scheduled rerolling of the per-lane continuous refresh.
//!
//! The refresh of a large network is mostly the same few assignment shapes
//! repeated per component instance (connection aliases, sign flips, branch
//! laws), interleaved component by component. Emitted one by one they make a
//! straight-line function whose size grows with the network, which both the
//! device compiler (ptxas) and the instruction fetch of a single GPU thread
//! pay for. This module reorders the refresh into dependency levels and, in
//! each level, renders the assignment programs of one shape as one loop over
//! tables of their `y`/`p` indices and constants.
//!
//! The reordering keeps every read-after-write, write-after-read and
//! write-after-write order between units (assignment programs and projection
//! block calls), and each program keeps its own operations, so every slot is
//! computed from the same inputs by the same arithmetic as in program order.
//! A unit whose `y` reads cannot be listed (an op with nested programs) is a
//! barrier: it stays after everything before it and before everything after.

use std::collections::{BTreeMap, BTreeSet};

use rumoca_ir_solve as solve;

/// One element of the refresh in program order.
pub(super) enum Unit {
    /// An assignment program lowered to scalar ops, storing to `targets`.
    Assign {
        ops: Vec<solve::LinearOp>,
        targets: Vec<usize>,
    },
    /// Anything rendered as before (a projection block call, or an
    /// assignment program outside the rerollable op subset), with its `y`
    /// footprint, or `None` when it must be a barrier.
    Opaque {
        id: usize,
        footprint: Option<(BTreeSet<usize>, BTreeSet<usize>)>,
    },
}

/// The `y` slots a program reads, or `None` if some op hides its reads.
pub(super) fn y_reads(ops: &[solve::LinearOp]) -> Option<BTreeSet<usize>> {
    let mut reads = BTreeSet::new();
    for op in ops {
        match op {
            solve::LinearOp::LoadY { index, .. } => {
                reads.insert(*index);
            }
            solve::LinearOp::TensorLoad {
                input: solve::TensorInputKind::Y,
                input_start,
                count,
                ..
            } => reads.extend(*input_start..input_start + count),
            solve::LinearOp::FunctionConditional { .. }
            | solve::LinearOp::FunctionFold { .. }
            | solve::LinearOp::GuardedFunctionFold { .. } => return None,
            _ => {}
        }
    }
    Some(reads)
}

/// One literal operand of a rerolled program, in op order.
#[derive(Clone, Copy, PartialEq)]
enum Literal {
    Y(usize),
    P(usize),
    Const(f64),
    Target(usize),
}

/// The op subset a rerolled loop body renders; anything else stays single.
fn literals_and_shape(ops: &[solve::LinearOp], targets: &[usize]) -> Option<(String, Vec<Literal>)> {
    use std::fmt::Write;
    let mut shape = String::new();
    let mut literals = Vec::new();
    let mut store = 0usize;
    for op in ops {
        match op {
            solve::LinearOp::Const { dst, value } => {
                write!(shape, "C{dst};").ok()?;
                literals.push(Literal::Const(*value));
            }
            solve::LinearOp::LoadTime { dst } => write!(shape, "T{dst};").ok()?,
            solve::LinearOp::LoadY { dst, index } => {
                write!(shape, "Y{dst};").ok()?;
                literals.push(Literal::Y(*index));
            }
            solve::LinearOp::LoadP { dst, index } => {
                write!(shape, "P{dst};").ok()?;
                literals.push(Literal::P(*index));
            }
            solve::LinearOp::Move { dst, src } => write!(shape, "M{dst},{src};").ok()?,
            solve::LinearOp::Unary { dst, op, arg } => write!(shape, "U{dst},{op:?},{arg};").ok()?,
            solve::LinearOp::Binary { dst, op, lhs, rhs } => {
                write!(shape, "B{dst},{op:?},{lhs},{rhs};").ok()?
            }
            solve::LinearOp::Compare { dst, op, lhs, rhs } => {
                write!(shape, "Q{dst},{op:?},{lhs},{rhs};").ok()?
            }
            solve::LinearOp::Select {
                dst,
                cond,
                if_true,
                if_false,
            } => write!(shape, "S{dst},{cond},{if_true},{if_false};").ok()?,
            solve::LinearOp::StoreOutput { src } => {
                write!(shape, "O{src};").ok()?;
                literals.push(Literal::Target(*targets.get(store)?));
                store += 1;
            }
            _ => return None,
        }
    }
    (store == targets.len()).then_some((shape, literals))
}

fn unary(op: &solve::UnaryOp, arg: &str) -> Option<String> {
    use solve::UnaryOp as U;
    Some(match op {
        U::Neg => format!("(-{arg})"),
        U::Not => format!("({arg} == 0.0 ? 1.0 : 0.0)"),
        U::Abs => format!("fabs({arg})"),
        U::Sign => format!("({arg} > 0.0 ? 1.0 : ({arg} < 0.0 ? -1.0 : 0.0))"),
        U::Sqrt => format!("sqrt({arg})"),
        U::Floor => format!("floor({arg})"),
        U::Ceil => format!("ceil({arg})"),
        U::Trunc => format!("trunc({arg})"),
        U::Sin => format!("sin({arg})"),
        U::Cos => format!("cos({arg})"),
        U::Tan => format!("tan({arg})"),
        U::Asin => format!("asin({arg})"),
        U::Acos => format!("acos({arg})"),
        U::Atan => format!("atan({arg})"),
        U::Sinh => format!("sinh({arg})"),
        U::Cosh => format!("cosh({arg})"),
        U::Tanh => format!("tanh({arg})"),
        U::Exp => format!("exp({arg})"),
        U::Log => format!("log({arg})"),
        U::Log10 => format!("log10({arg})"),
        #[allow(unreachable_patterns)]
        _ => return None,
    })
}

fn binary(op: &solve::BinaryOp, l: &str, r: &str) -> Option<String> {
    use solve::BinaryOp as B;
    Some(match op {
        B::Add => format!("({l} + {r})"),
        B::Sub => format!("({l} - {r})"),
        B::Mul => format!("({l} * {r})"),
        B::Div => format!("({l} / {r})"),
        B::Pow => format!("pow({l}, {r})"),
        B::And => format!("(({l} != 0.0 && {r} != 0.0) ? 1.0 : 0.0)"),
        B::Or => format!("(({l} != 0.0 || {r} != 0.0) ? 1.0 : 0.0)"),
        B::Atan2 => format!("atan2({l}, {r})"),
        B::Min => format!("rumoca_real_min({l}, {r})"),
        B::Max => format!("rumoca_real_max({l}, {r})"),
        #[allow(unreachable_patterns)]
        _ => return None,
    })
}

fn compare(op: &solve::CompareOp, l: &str, r: &str) -> String {
    use solve::CompareOp as C;
    let symbol = match op {
        C::Lt => "<",
        C::Le => "<=",
        C::Gt => ">",
        C::Ge => ">=",
        C::Eq => "==",
        C::Ne => "!=",
    };
    format!("({l} {symbol} {r} ? 1.0 : 0.0)")
}

/// A C double literal for a table, exact for every finite value.
fn real_literal(value: f64) -> String {
    if value.is_nan() {
        "NAN".to_string()
    } else if value.is_infinite() {
        if value > 0.0 { "INFINITY".to_string() } else { "(-INFINITY)".to_string() }
    } else {
        // `{:?}` prints the shortest string that round-trips to the same
        // binary64 value; force a decimal point so C reads a double.
        let text = format!("{value:?}");
        if text.contains(['.', 'e', 'E']) { text } else { format!("{text}.0") }
    }
}

/// The loop rendering one group of `instances` programs of one shape.
struct Group {
    ops: Vec<solve::LinearOp>,
    instances: Vec<Vec<Literal>>,
}

impl Group {
    /// C code of the loop, with its tables named `<prefix>_<id>i/_r`.
    fn render(&self, prefix: &str, id: usize, tables: &mut String) -> Option<String> {
        use std::fmt::Write;
        let first = self.instances.first()?;
        let ints: Vec<usize> = first
            .iter()
            .enumerate()
            .filter(|(_, l)| !matches!(l, Literal::Const(_)))
            .map(|(k, _)| k)
            .collect();
        let reals: Vec<usize> = first
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l, Literal::Const(_)))
            .map(|(k, _)| k)
            .collect();
        let n = self.instances.len();
        let (ni, nr) = (ints.len(), reals.len());
        if ni > 0 {
            write!(tables, "__device__ static const int {prefix}_{id}i[{}] = {{", n * ni).ok()?;
            for (i, inst) in self.instances.iter().enumerate() {
                for &k in &ints {
                    let v = match inst[k] {
                        Literal::Y(v) | Literal::P(v) | Literal::Target(v) => v,
                        Literal::Const(_) => return None,
                    };
                    write!(tables, "{}{v}", if i == 0 && k == ints[0] { "" } else { "," }).ok()?;
                }
            }
            tables.push_str("};\n");
        }
        if nr > 0 {
            write!(tables, "__device__ static const double {prefix}_{id}r[{}] = {{", n * nr).ok()?;
            for (i, inst) in self.instances.iter().enumerate() {
                for &k in &reals {
                    let Literal::Const(v) = inst[k] else { return None };
                    write!(tables, "{}{}", if i == 0 && k == reals[0] { "" } else { "," }, real_literal(v)).ok()?;
                }
            }
            tables.push_str("};\n");
        }
        let column = |k: usize| -> String {
            if let Some(j) = ints.iter().position(|x| *x == k) {
                format!("{prefix}_{id}i[__k * {ni} + {j}]")
            } else {
                let j = reals.iter().position(|x| *x == k).unwrap_or(0);
                format!("{prefix}_{id}r[__k * {nr} + {j}]")
            }
        };
        let reg = |r: &solve::Reg| format!("__r{r}");
        let mut body = String::new();
        let mut literal = 0usize;
        let mut dsts = Vec::new();
        for op in &self.ops {
            let line = match op {
                solve::LinearOp::Const { dst, .. } => {
                    literal += 1;
                    dsts.push(reg(dst));
                    format!("const double {} = {};", reg(dst), column(literal - 1))
                }
                solve::LinearOp::LoadTime { dst } => {
                    dsts.push(reg(dst));
                    format!("const double {} = time;", reg(dst))
                }
                solve::LinearOp::LoadY { dst, .. } => {
                    literal += 1;
                    dsts.push(reg(dst));
                    format!("const double {} = batch_y[{}];", reg(dst), column(literal - 1))
                }
                solve::LinearOp::LoadP { dst, .. } => {
                    literal += 1;
                    dsts.push(reg(dst));
                    format!("const double {} = batch_p[{}];", reg(dst), column(literal - 1))
                }
                solve::LinearOp::Move { dst, src } => {
                    dsts.push(reg(dst));
                    format!("const double {} = {};", reg(dst), reg(src))
                }
                solve::LinearOp::Unary { dst, op, arg } => {
                    dsts.push(reg(dst));
                    format!("const double {} = {};", reg(dst), unary(op, &reg(arg))?)
                }
                solve::LinearOp::Binary { dst, op, lhs, rhs } => {
                    dsts.push(reg(dst));
                    format!("const double {} = {};", reg(dst), binary(op, &reg(lhs), &reg(rhs))?)
                }
                solve::LinearOp::Compare { dst, op, lhs, rhs } => {
                    dsts.push(reg(dst));
                    format!("const double {} = {};", reg(dst), compare(op, &reg(lhs), &reg(rhs)))
                }
                solve::LinearOp::Select {
                    dst,
                    cond,
                    if_true,
                    if_false,
                } => {
                    dsts.push(reg(dst));
                    format!(
                        "const double {} = ({} != 0.0 ? {} : {});",
                        reg(dst),
                        reg(cond),
                        reg(if_true),
                        reg(if_false)
                    )
                }
                solve::LinearOp::StoreOutput { src } => {
                    literal += 1;
                    format!("batch_y[{}] = {};", column(literal - 1), reg(src))
                }
                _ => return None,
            };
            body.push_str("        ");
            body.push_str(&line);
            body.push('\n');
        }
        for d in &dsts {
            writeln!(body, "        (void){d};").ok()?;
        }
        Some(format!(
            "    // {n} instances of one assignment shape\n    for (int __k = 0; __k < {n}; ++__k) {{\n{body}    }}\n"
        ))
    }
}

/// One step of the rerolled refresh.
pub(super) enum Step {
    /// The opaque unit with this id, rendered as before.
    Opaque(usize),
    /// A single assignment program rendered as before (index into units).
    Single(usize),
    /// A rerolled loop (its C code).
    Rolled(String),
}

/// Levels of the units: each unit one level after the deepest unit it must
/// follow (RAW, WAR, WAW on `y` slots); a barrier follows everything before
/// it and everything after it follows the barrier.
fn levels(units: &[Unit]) -> Vec<usize> {
    let mut level = Vec::with_capacity(units.len());
    let mut last_write: BTreeMap<usize, usize> = BTreeMap::new();
    let mut readers: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut floor = 0usize; // first level after the last barrier
    let mut deepest = 0usize;
    for (u, unit) in units.iter().enumerate() {
        let footprint = match unit {
            Unit::Assign { ops, targets } => {
                y_reads(ops).map(|reads| (targets.iter().copied().collect::<BTreeSet<_>>(), reads))
            }
            Unit::Opaque { footprint, .. } => footprint.clone(),
        };
        let Some((writes, reads)) = footprint else {
            let l = if u == 0 { 0 } else { deepest + 1 };
            level.push(l);
            floor = l + 1;
            deepest = l;
            last_write.clear();
            readers.clear();
            continue;
        };
        let mut l = floor;
        for v in reads.iter().chain(writes.iter()) {
            if let Some(&w) = last_write.get(v) {
                l = l.max(level[w] + 1);
            }
        }
        for v in &writes {
            if let Some(rs) = readers.get(v) {
                for &r in rs {
                    l = l.max(level[r] + 1);
                }
            }
        }
        level.push(l);
        deepest = deepest.max(l);
        for v in &writes {
            last_write.insert(*v, u);
            readers.remove(v);
        }
        for v in reads.difference(&writes) {
            readers.entry(*v).or_default().push(u);
        }
    }
    level
}

/// The rerolled refresh: steps in level order and the tables they read.
/// Groups smaller than `min_group` stay single programs.
pub(super) fn reroll(units: &[Unit], prefix: &str, min_group: usize) -> (Vec<Step>, String) {
    let level = levels(units);
    let depth = level.iter().copied().max().map_or(0, |d| d + 1);
    let mut steps = Vec::new();
    let mut tables = String::new();
    let mut next_id = 0usize;
    for l in 0..depth {
        // Shapes in first-appearance order keep the output deterministic.
        let mut order: Vec<String> = Vec::new();
        let mut groups: BTreeMap<String, (Vec<solve::LinearOp>, Vec<(usize, Vec<Literal>)>)> = BTreeMap::new();
        for (u, unit) in units.iter().enumerate() {
            if level[u] != l {
                continue;
            }
            match unit {
                Unit::Opaque { id, .. } => steps.push(Step::Opaque(*id)),
                Unit::Assign { ops, targets } => match literals_and_shape(ops, targets) {
                    Some((shape, literals)) => {
                        let entry = groups.entry(shape.clone()).or_insert_with(|| {
                            order.push(shape.clone());
                            (ops.clone(), Vec::new())
                        });
                        entry.1.push((u, literals));
                    }
                    None => steps.push(Step::Single(u)),
                },
            }
        }
        for shape in order {
            let (ops, members) = groups.remove(&shape).unwrap_or_default();
            if members.len() < min_group {
                steps.extend(members.iter().map(|(u, _)| Step::Single(*u)));
                continue;
            }
            let group = Group {
                ops,
                instances: members.iter().map(|(_, lits)| lits.clone()).collect(),
            };
            match group.render(prefix, next_id, &mut tables) {
                Some(code) => {
                    steps.push(Step::Rolled(code));
                    next_id += 1;
                }
                None => steps.extend(members.iter().map(|(u, _)| Step::Single(*u))),
            }
        }
    }
    (steps, tables)
}
