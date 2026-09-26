use eqiora_compiler::source_identity::LocalSourceIdentity;
use eqiora_lang::{format, parse};

#[test]
fn notation_is_presentation_not_local_source_semantic_identity() {
    let plain = "model Main() { parameter viscosity: 1 = 2; variable stress: 1; relation law { stress = viscosity; } }";
    let decorated = plain
        .replace("Main()", r"Main @{\mathcal{M}}()")
        .replace("viscosity:", r"viscosity @{\mu}:")
        .replace("stress:", r"stress @{\sigma_{ij}}:")
        .replace("law {", r"law @{L} {");
    let original = parse("plain.eqi", plain).into_document().unwrap();
    let edited = parse("edited.eqi", &decorated).into_document().unwrap();
    assert_ne!(format(&original), format(&edited));
    assert_eq!(
        LocalSourceIdentity::from_document(&original).unwrap(),
        LocalSourceIdentity::from_document(&edited).unwrap()
    );
    let renamed = parse("renamed.eqi", &decorated.replace("stress", "traction"))
        .into_document()
        .unwrap();
    assert_ne!(
        LocalSourceIdentity::from_document(&edited).unwrap(),
        LocalSourceIdentity::from_document(&renamed).unwrap()
    );
}

#[test]
fn activation_token_locations_do_not_enter_local_source_semantic_identity() {
    let plain = "model M(){clock tick=periodic(1[s]);state memory:1 at tick;port out:signal output 1 at tick;let previous at tick=pre(memory);relation update at tick{next(memory)=previous;out=memory;}}";
    let edited = plain.replace(" at tick", " at // 🧪 moved token\r\n tick");
    let original = parse("plain.eqi", plain).into_document().unwrap();
    let relocated = parse("relocated.eqi", &edited).into_document().unwrap();
    assert_eq!(
        LocalSourceIdentity::from_document(&original).unwrap(),
        LocalSourceIdentity::from_document(&relocated).unwrap()
    );
    let renamed = parse("renamed.eqi", &edited.replace("tick", "tock"))
        .into_document()
        .unwrap();
    assert_ne!(
        LocalSourceIdentity::from_document(&original).unwrap(),
        LocalSourceIdentity::from_document(&renamed).unwrap()
    );
}
