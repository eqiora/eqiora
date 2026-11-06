//! Analytic geometry expectations frozen before graph/Python implementation.
use eqiora_geometry::{CanonicalGeometryV1, NamedEntitySet, PlanarFace, PlanarRegion};

fn skew_polygon() -> Vec<[f64;2]> {
    vec![[0.25,-0.5],[1.375,1.0],[0.0625,5.5],[-1.0625,4.0]]
}

fn skew_polyhedron() -> (Vec<[f64;3]>,Vec<Vec<usize>>) {
    let vertices = (0..8).map(|n| {
        let q=[f64::from(n&1),f64::from((n>>1)&1),f64::from((n>>2)&1)];
        [0.25+15./16.*q[0]-3./32.*q[1]+11./64.*q[2],
         -0.5+5./4.*q[0]+3.*q[1]-51./32.*q[2],
         0.75+5./2.*q[1]+15./4.*q[2]]
    }).collect();
    (vertices,vec![vec![0,4,6,2],vec![1,3,7,5],vec![0,1,5,4],
                  vec![2,6,7,3],vec![0,2,3,1],vec![4,5,7,6]])
}

#[test]
fn analytic_skew_domains_use_existing_exact_geometry_families() {
    let v=skew_polygon();
    let twice_area:f64=(0..4).map(|i|v[i][0]*v[(i+1)%4][1]-v[(i+1)%4][0]*v[i][1]).sum();
    assert_eq!(twice_area/2.,225./32.);
    let region=PlanarRegion::new(v,vec![PlanarFace::new(vec![0,1,2,3],vec![])],
        vec![NamedEntitySet::new("domain",2,vec![0])],1e-12).unwrap();
    assert_eq!(region.vertices().len(),4);
    let (v,f)=skew_polyhedron();
    let a=[15./16.,5./4.,0.]; let b=[-3./32.,3.,5./2.]; let c=[11./64.,-51./32.,15./4.];
    let determinant=a[0]*(b[1]*c[2]-b[2]*c[1])-b[0]*(a[1]*c[2]-a[2]*c[1])+c[0]*(a[1]*b[2]-a[2]*b[1]);
    assert_eq!(determinant,15625./1024.);
    let geometry=CanonicalGeometryV1::from_convex_polyhedra(v,vec![f],vec![NamedEntitySet::new("domain",3,vec![0])],1e-12).unwrap();
    assert_eq!(geometry.polyhedral_vertices().unwrap().len(),8);
}
