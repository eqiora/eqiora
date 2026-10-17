use super::super::super::without_draws;
use super::super::tests::{POISSON, WIENER, request};
use super::*;
use crate::{CommonTimeCheckpoint, RecordedTimeCheckpoint};

#[test]
fn canonical_checkpoint_records_reopen_without_drawing_and_restore_explicitly() {
    for (source, method) in [
        (WIENER, ProcessMethod::EulerMaruyama { depth: 8 }),
        (POISSON, ProcessMethod::PoissonEvents { max_events: 64 }),
    ] {
        let run = request(source, method);
        let mut cursor = run.start();
        cursor.accept(cursor.trial(&exact(0.5)).unwrap()).unwrap();
        let checkpoint = CommonTimeCheckpoint::from_process(cursor).unwrap();
        let record = without_draws(|| checkpoint.record()).unwrap();
        assert_eq!(record.identity(), checkpoint.identity());
        assert_eq!(record.request(), checkpoint.request());
        let reopened = without_draws(|| {
            RecordedTimeCheckpoint::from_bytes(record.as_bytes(), run.request.plan())
        })
        .unwrap();
        assert_eq!(reopened, record);
        assert_eq!(reopened.restore().unwrap(), checkpoint);
        // The guard must detect the path reconstruction used by true restart.
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                without_draws(|| reopened.restore().unwrap())
            }))
            .is_err()
        );
        for end in [0, record.as_bytes().len() / 2, record.as_bytes().len() - 1] {
            assert!(
                RecordedTimeCheckpoint::from_bytes(&record.as_bytes()[..end], run.request.plan())
                    .is_err()
            );
        }
        let mut trailing = record.as_bytes().to_vec();
        trailing.push(b'\n');
        assert!(RecordedTimeCheckpoint::from_bytes(&trailing, run.request.plan()).is_err());
        let mut corrupt: serde_json::Value = serde_json::from_slice(record.as_bytes()).unwrap();
        corrupt["identity"] = serde_json::json!("foreign");
        assert!(
            RecordedTimeCheckpoint::from_bytes(
                &serde_json::to_vec(&corrupt).unwrap(),
                run.request.plan()
            )
            .is_err()
        );
        let foreign = request(
            &WIENER.replace("-x * 2", "-x * 3"),
            ProcessMethod::EulerMaruyama { depth: 8 },
        );
        assert!(
            RecordedTimeCheckpoint::from_bytes(record.as_bytes(), foreign.request.plan()).is_err()
        );
    }
}

#[test]
fn inspecting_canonical_content_does_not_grant_physical_or_restart_authority() {
    let run = request(WIENER, ProcessMethod::EulerMaruyama { depth: 8 });
    let mut cursor = run.start();
    cursor.accept(cursor.trial(&exact(0.5)).unwrap()).unwrap();
    let bytes = cursor.checkpoint_bytes().unwrap();
    let mut payload: Payload = checkpoint::decode(&bytes, PROFILE, Ok).unwrap();
    payload.state_base64 = STANDARD.encode(
        run.program()
            .accept_state(0.5, vec![123.], "result")
            .unwrap()
            .to_bytes()
            .unwrap(),
    );
    let reidentified = checkpoint::encode(PROFILE, &payload).unwrap();
    // This is canonical recorded content, but no longer an admissible restart.
    // The record exposes no accepted State or controller; restoration cannot
    // bypass the original physical/path validator merely because it was viewed.
    let record =
        without_draws(|| RecordedTimeCheckpoint::from_bytes(&reidentified, run.request.plan()))
            .unwrap();
    assert_eq!(record.as_bytes(), reidentified);
    let error = record.restore().unwrap_err();
    assert!(
        error.message().contains("physical State differs"),
        "{error:?}"
    );
}
