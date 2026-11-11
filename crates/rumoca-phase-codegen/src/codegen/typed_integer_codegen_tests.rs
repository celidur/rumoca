//! Executable checks for typed integer arithmetic emitted by the pure-call template.

use super::*;
use std::fs;
use std::process::Command;

fn integer_type(minimum: i64, maximum: i64) -> Value {
    Value::from_serialize(serde_json::json!({
        "scalar": {
            "kind": "integer",
            "profile": { "minimum": minimum, "maximum": maximum }
        },
        "scalar_count": 1
    }))
}

fn render_arithmetic_harness() -> String {
    let mut env = create_environment();
    env.add_template(
        "typed_arithmetic",
        include_str!("../templates/fmi3/typed_arithmetic.jinja"),
    )
    .expect("typed arithmetic template parses");
    env.add_template(
        "harness",
        r#"{% import "typed_arithmetic" as arithmetic %}
static int checked_full(int64_t a, int64_t b, int64_t out[1]) {
    {{ arithmetic.integer_binary("integer_quotient", "out[0]", "a", "b") }}
    {{ arithmetic.domain_check("out", full_type) }}
    return 0;
}
static int checked_small(int64_t a, int64_t b, int64_t out[1]) {
    {{ arithmetic.integer_binary("integer_quotient", "out[0]", "a", "b") }}
    {{ arithmetic.domain_check("out", small_type) }}
    return 0;
}
"#,
    )
    .expect("test harness template parses");
    env.get_template("harness")
        .expect("test harness is registered")
        .render(minijinja::context! {
            full_type => integer_type(i64::MIN, i64::MAX),
            small_type => integer_type(-5, 5),
        })
        .expect("test harness renders")
}

#[test]
fn integer_quotient_matches_checked_truncating_division_in_c99() {
    let directory = tempfile::tempdir().expect("temporary test directory");
    let source = directory.path().join("integer_quotient.c");
    let executable = directory.path().join("integer_quotient");
    let mut program =
        String::from("#include <stdint.h>\n#include <stddef.h>\n#include <limits.h>\n");
    program.push_str(&render_arithmetic_harness());
    program.push_str(
        r#"
static int expect(int (*operation)(int64_t, int64_t, int64_t *),
                  int64_t lhs, int64_t rhs, int expected_status,
                  int64_t expected_value) {
    int64_t result = INT64_C(123);
    int status = operation(lhs, rhs, &result);
    return status == expected_status &&
           (expected_status != 0 || result == expected_value);
}

int main(void) {
    if (!expect(checked_full, 7, 3, 0, 2)) return 1;
    if (!expect(checked_full, 7, -3, 0, -2)) return 2;
    if (!expect(checked_full, -7, 3, 0, -2)) return 3;
    if (!expect(checked_full, -7, -3, 0, 2)) return 4;
    if (!expect(checked_full, 0, -3, 0, 0)) return 5;
    if (!expect(checked_full, INT64_MIN, 1, 0, INT64_MIN)) return 6;
    if (!expect(checked_full, INT64_MAX, 1, 0, INT64_MAX)) return 7;
    if (!expect(checked_full, INT64_MIN, -1, 1, INT64_C(123))) return 8;
    if (!expect(checked_full, 1, 0, 1, INT64_C(123))) return 9;
    if (!expect(checked_small, 10, 2, 0, 5)) return 10;
    if (!expect(checked_small, 12, 2, 1, INT64_C(123))) return 11;
    return 0;
}
"#,
    );
    fs::write(&source, program).expect("write C harness");

    let compile = Command::new("cc")
        .arg("-std=c99")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-O2")
        .arg(&source)
        .arg("-o")
        .arg(&executable)
        .output()
        .expect("C99 compiler is available for executable codegen test");
    assert!(
        compile.status.success(),
        "generated C99 should compile:\n{}\n{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&executable)
        .output()
        .expect("run generated integer quotient checks");
    assert!(
        run.status.success(),
        "generated integer quotient checks failed with {:?}",
        run.status.code()
    );
}
