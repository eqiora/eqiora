//! Apply current document overlays only to sources admitted by the package owner.
use super::*;

pub(super) fn analyze(
    group: &str,
    version: u64,
    documents: &[AnalysisDocument],
    mut project: PackageProject,
    cancelled: &AtomicBool,
) -> Option<AnalysisOutcome> {
    // Newly discovered files are absent from the previous URI map. Reuse the
    // native package admission result once to apply their open buffers before
    // publishing. If the graph changes again, fall back to current open text.
    for _ in 0..2 {
        if cancelled.load(Ordering::Acquire) {
            return Some(AnalysisOutcome::Cancelled);
        }
        let overrides = documents
            .iter()
            .filter_map(|document| {
                project
                    .relative_by_uri
                    .get(&document.key)
                    .cloned()
                    .map(|path| (path, document.source.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        let (snapshot, paths) = EditorWorkspaceSnapshot::analyze_local_package_project_v1(
            version,
            &project.root_path,
            &overrides,
        )
        .ok()?;
        if cancelled.load(Ordering::Acquire) {
            return Some(AnalysisOutcome::Cancelled);
        }
        let (analysis, relative_by_uri) = package_workspace(group, snapshot, paths)?;
        let missing_overlay = documents.iter().any(|document| {
            relative_by_uri
                .get(&document.key)
                .is_some_and(|path| overrides.get(path) != Some(&document.source))
        });
        if missing_overlay {
            project.relative_by_uri = relative_by_uri;
            continue;
        }
        return Some(AnalysisOutcome::Workspace {
            analysis,
            relative_by_uri: Some(relative_by_uri),
        });
    }
    None
}
