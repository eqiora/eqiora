use eqiora_numerics::AuthenticatedCommonMesh;
use eqiora_artifact::GmshMeshPolicyV1;
use eqiora_geometry::{CanonicalGeometryV1, PlanarRegion, PlanarFace, NamedEntitySet};
fn main() {
    let source = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n$Elements\n1 2 1 2\n2 1 2 2\n1 1 2 3\n2 1 3 4\n$EndElements\n";
    let geometry = CanonicalGeometryV1::from_region(&PlanarRegion::new(
        vec![[0.,0.],[1.,0.],[1.,1.],[0.,1.]], vec![PlanarFace::new(vec![0,1,2,3],vec![])],
        vec![NamedEntitySet::new("body",2,vec![0]),NamedEntitySet::new("outer",1,vec![0,1,2,3])],1e-12).unwrap()).unwrap();
    let owner = AuthenticatedCommonMesh::gmsh_4152(geometry.clone(),GmshMeshPolicyV1::explicit(1e-12,0.01,8,5.).unwrap(),source.to_vec()).unwrap();
    let output = std::path::PathBuf::from(std::env::var("ARCHIVE_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    std::fs::write(output.join("commonmesh-v4-gmsh4152.json"),owner.to_bytes().unwrap()).unwrap();
    std::fs::write(output.join("lineage-v1-gmsh4152.json"),owner.production().unwrap().canonical_json().unwrap()).unwrap();
    let mut grid = String::from("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n1 16 1 16\n2 1 0 16\n");
    for tag in 1..=16 { grid += &format!("{tag}\n"); }
    for y in 0..4 { for x in 0..4 { grid += &format!("{} {} 0\n",x as f64/3.,y as f64/3.); } }
    grid += "$EndNodes\n$Elements\n1 18 1 18\n2 1 2 18\n";
    let mut tag = 1;
    for y in 0..3 { for x in 0..3 {
        let a = y*4+x+1;
        grid += &format!("{tag} {a} {} {}\n",a+1,a+5); tag += 1;
        grid += &format!("{tag} {a} {} {}\n",a+5,a+4); tag += 1;
    }}
    grid += "$EndElements\n";
    let legacy = AuthenticatedCommonMesh::gmsh_4152(geometry,GmshMeshPolicyV1::explicit(1e-12,0.01,8,5.).unwrap(),grid.into_bytes()).unwrap();
    std::fs::write(output.join("commonmesh-v4-planar-over-facet-policy.json"),legacy.to_bytes().unwrap()).unwrap();

    let original = String::from_utf8(owner.to_bytes().unwrap()).unwrap();
    let new_kind = original.replace("\"kind\":\"gmsh4152\"", "\"kind\":\"external-msh41\"");
    assert_ne!(new_kind, original);
    let rejected = AuthenticatedCommonMesh::from_bytes(new_kind.as_bytes()).unwrap_err();
    assert!(rejected.message().contains("unknown variant `external-msh41`"));
    println!("archived-214: original v4 reader rejects external-msh41 discriminator");

}
