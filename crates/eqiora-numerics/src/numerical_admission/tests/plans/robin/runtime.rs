//! Repository identity and immutable home-backed scratch for the ordinary test.
use crate::numerical_admission::tests::artifact_capture::{digest, json_bytes, write};
use std::path::{Path, PathBuf};
use std::process::Command;

pub(super) fn output_root() -> PathBuf {
    let home = PathBuf::from(std::env::var_os("HOME").expect("HOME is required for test scratch"))
        .canonicalize()
        .unwrap();
    if let Some(explicit) = std::env::var_os("EQIORA_ROBIN_OUTPUT") {
        let root = PathBuf::from(explicit);
        assert!(root.is_absolute() && root.starts_with(&home));
        std::fs::create_dir_all(root.parent().unwrap()).unwrap();
        std::fs::create_dir(&root).expect("fresh Robin output required; preserve earlier attempts");
        return root;
    }
    let parent = home.join(".cache/eqiora/robin-correctness");
    std::fs::create_dir_all(&parent).unwrap();
    for _ in 0..64 {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = parent.join(format!("{}-{nonce}", std::process::id()));
        match std::fs::create_dir(&root) {
            Ok(()) => return root,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("cannot create home-backed Robin scratch: {error}"),
        }
    }
    panic!("cannot allocate a fresh Robin capture directory without overwriting attempts")
}

pub(super) fn source_identity(root: &Path) -> String {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let git = |arguments: &[&str]| {
        let output = Command::new("git")
            .current_dir(&repository)
            .args(arguments)
            .output()
            .expect("Robin evidence requires git to identify the actual tested checkout");
        assert!(output.status.success(), "git source identity failed");
        output.stdout
    };
    let head = String::from_utf8(git(&["rev-parse", "HEAD"])).unwrap();
    let head = head.trim().to_owned();
    let tree = String::from_utf8(git(&["rev-parse", "HEAD^{tree}"])).unwrap();
    let status = git(&["status", "--porcelain"]);
    let difference = git(&["diff", "--binary", "HEAD"]);
    if let Ok(expected) = std::env::var("EQIORA_ROBIN_SOURCE_HEAD") {
        assert_eq!(expected, head, "explicit producer identity is stale");
    }
    // A dirty checkout is recorded as head plus its actual diff, never as a clean tree.
    write(&root.join("source-working.diff"), &difference);
    write(
        &root.join("source-identity.json"),
        &json_bytes(&serde_json::json!({
            "git_head":head,"git_head_tree":tree.trim(),
            "working_tree_clean":status.is_empty(),
            "working_tree_status":String::from_utf8(status).unwrap(),
            "tracked_working_diff_sha256":digest(&difference),
        })),
    );
    head
}
