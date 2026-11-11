//! MLS §9.2 (SPEC_0022 CONN-005): connected variables with non-empty
//! `quantity` attributes must have equal quantities. The opt-in
//! `ConnectorQuantityPolicy::Warn` reports a mismatch as an EF037 warning and
//! keeps the connection.

use rumoca_ir_ast as ast;
use rumoca_phase_flatten::{
    ConnectorQuantityPolicy, ConnectorQuantityWarning, FlattenOptions,
    flatten_ref_with_connector_quantity_policy,
};

const SOURCE: &str = r#"
connector VoltageIn = input Real(quantity = "Voltage", unit = "V");
connector PotentialOut = output Real(quantity = "ElectricPotential", unit = "V");
model Source
  PotentialOut y;
equation
  y = 1;
end Source;
model Sink
  VoltageIn u;
  Real z;
equation
  z = u;
end Sink;
model Probe
  Source source;
  Sink sink;
equation
  connect(source.y, sink.u);
end Probe;
"#;

fn instanced(source: &str, model: &str) -> (ast::ClassTree, ast::InstanceOverlay) {
    let file_name = "<connector_quantity_policy>";
    let stored = rumoca_phase_parse::parse_to_ast(source, file_name).expect("source parses");
    let mut tree = ast::ClassTree::from_parsed(stored);
    tree.source_map.add(file_name, source);
    let resolved =
        rumoca_phase_resolve::resolve(ast::ParsedTree::new(tree)).expect("source resolves");
    let instanced =
        rumoca_phase_instantiate::instantiate(resolved, model).expect("model instantiates");
    let ast::InstancedTree { tree, mut overlay } = instanced;
    rumoca_phase_typecheck::typecheck_instanced(&tree, &mut overlay, model)
        .expect("instanced model typechecks");
    (tree, overlay)
}

#[test]
fn strict_policy_refuses_a_scalar_connector_quantity_mismatch() {
    let (tree, overlay) = instanced(SOURCE, "Probe");
    assert!(rumoca_phase_flatten::flatten_ref(&tree, &overlay, "Probe").is_err());
    let (result, warnings) = flatten_ref_with_connector_quantity_policy(
        &tree,
        &overlay,
        "Probe",
        FlattenOptions::default(),
        ConnectorQuantityPolicy::default(),
    );
    assert!(result.is_err());
    assert!(warnings.is_empty());
}

#[test]
fn warn_policy_reports_a_scalar_connector_quantity_mismatch() {
    let (tree, overlay) = instanced(SOURCE, "Probe");
    let (result, warnings) = flatten_ref_with_connector_quantity_policy(
        &tree,
        &overlay,
        "Probe",
        FlattenOptions::default(),
        ConnectorQuantityPolicy::Warn,
    );
    result.expect("the warning policy keeps the connection");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    let warning = &warnings[0];
    assert_eq!(
        (warning.variable_a.as_str(), warning.variable_b.as_str()),
        ("source.y", "sink.u")
    );
    assert_eq!(
        (warning.quantity_a.as_str(), warning.quantity_b.as_str()),
        ("ElectricPotential", "Voltage")
    );
    assert_eq!(warning.unit_a.as_deref(), Some("V"));
    assert_eq!(warning.unit_b.as_deref(), Some("V"));
    let diagnostic = warning.diagnostic();
    assert_eq!(
        diagnostic.code.as_deref(),
        Some(ConnectorQuantityWarning::CODE)
    );
    assert_eq!(
        diagnostic.severity,
        rumoca_core::DiagnosticSeverity::Warning
    );
    assert_eq!(diagnostic.labels.len(), 3);
}
