use super::*;

/// A parsed in-memory source whose syntax and source text were produced
/// together.
///
/// Keeping the text beside the parsed definition preserves the parser's stable
/// source identity through session assembly and into provenance-bearing IR.
#[derive(Debug, Clone)]
pub struct ParsedSourceDocument {
    uri: String,
    source: Arc<str>,
    definition: ast::StoredDefinition,
}

impl ParsedSourceDocument {
    pub fn parse(uri: impl Into<String>, source: impl Into<String>) -> Result<Self> {
        let uri = uri.into();
        let text = source.into();
        // The retained text is the expanded text, so the spans the parse mints
        // are offsets into the text this document carries.
        let source = Arc::<str>::from(
            crate::parse::expanded_document_source(&text, &uri)
                .map_err(anyhow::Error::new)?
                .into_owned(),
        );
        let definition = crate::parse::parse_source_to_ast(&source, &uri)?;
        Ok(Self {
            uri,
            source,
            definition,
        })
    }

    /// Parse a document using an explicit string-escape compatibility policy.
    ///
    /// As with [`Self::parse`], extension expansion happens before the source
    /// text and AST are retained together. Warning spans therefore address
    /// the exact text stored by this document, and `uri` remains the parser's
    /// source identity.
    pub fn parse_with_string_escape_policy(
        uri: impl Into<String>,
        source: impl Into<String>,
        policy: rumoca_phase_parse::StringEscapePolicy,
    ) -> Result<(Self, Vec<rumoca_phase_parse::StringEscapeWarning>)> {
        let uri = uri.into();
        let text = source.into();
        if policy == rumoca_phase_parse::StringEscapePolicy::Strict {
            return Self::parse(uri, text).map(|document| (document, Vec::new()));
        }
        let source = Arc::<str>::from(
            crate::parse::expanded_document_source(&text, &uri)
                .map_err(anyhow::Error::new)?
                .into_owned(),
        );
        let (definition, warnings) =
            rumoca_phase_parse::parse_to_ast_with_string_escape_policy(&source, &uri, policy)
                .map_err(|errors| anyhow::anyhow!("parse {uri}: {errors:?}"))?;
        Ok((
            Self {
                uri,
                source,
                definition,
            },
            warnings,
        ))
    }

    pub fn uri(&self) -> &str {
        &self.uri
    }

    pub(super) fn from_parsed(
        uri: String,
        source: Arc<str>,
        definition: ast::StoredDefinition,
    ) -> Self {
        Self {
            uri,
            source,
            definition,
        }
    }

    pub(super) fn into_parts(self) -> (String, Arc<str>, ast::StoredDefinition) {
        (self.uri, self.source, self.definition)
    }
}

#[derive(Debug, Clone)]
pub struct ParsedSourceRootLoad<'a> {
    pub source_root_kind: SourceRootKind,
    pub source_root_path: &'a Path,
    pub cache_status: SourceRootCacheStatus,
    pub path_key: &'a str,
    pub current_document_path: Option<&'a str>,
    pub documents: Vec<(String, ast::StoredDefinition)>,
    pub expected_epoch: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_string_policy_keeps_raw_document_uri_and_text() {
        let uri = "file:///tmp/RawEscape.mo";
        let source = "model RawEscape\n  parameter String label = \"café\\q\";\nend RawEscape;\n";
        let escape_start = source.find("\\q").expect("unknown escape");
        let (document, warnings) = ParsedSourceDocument::parse_with_string_escape_policy(
            uri,
            source,
            rumoca_phase_parse::StringEscapePolicy::OpenModelica,
        )
        .expect("OpenModelica policy accepts unknown string escape");

        assert_eq!(document.uri(), uri);
        assert_eq!(document.source.as_ref(), source);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].file_name, uri);
        assert_eq!(warnings[0].byte_start, escape_start);
        assert_eq!(warnings[0].byte_end, escape_start + 2);
        assert_eq!(warnings[0].escape, "\\q");
        assert!(ParsedSourceDocument::parse(uri, source).is_err());
        assert!(
            ParsedSourceDocument::parse_with_string_escape_policy(
                uri,
                source,
                rumoca_phase_parse::StringEscapePolicy::Strict,
            )
            .is_err()
        );
    }

    #[test]
    fn explicit_strict_policy_matches_existing_document_parser() {
        let uri = "file:///tmp/StrictDocument.mo";
        let source =
            "model StrictDocument\n  parameter String label = \"line\\n\";\nend StrictDocument;\n";
        let original = ParsedSourceDocument::parse(uri, source).expect("existing strict parser");
        let (explicit, warnings) = ParsedSourceDocument::parse_with_string_escape_policy(
            uri,
            source,
            rumoca_phase_parse::StringEscapePolicy::Strict,
        )
        .expect("explicit strict parser");
        assert!(warnings.is_empty());
        assert_eq!(explicit.uri, original.uri);
        assert_eq!(explicit.source, original.source);
        assert_eq!(
            format!("{:?}", explicit.definition),
            format!("{:?}", original.definition)
        );
    }
}
