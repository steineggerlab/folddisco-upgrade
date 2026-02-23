fn evalue_fitting(x: f32, m: f32, l: f32) -> f32 {
    // x: score, m: index size, l: query residue length 
    let x_d = x as f64;
    let m_d = m as f64;
    let l_d = l as f64;

    let mu = 10.09;
    let lam = -0.0034101279543267884 * l_d + 0.2727158726147608;
    
    let y = lam * (x_d - mu);

    let t = (-y).exp();
    let p_val = 1.0 - (-t).exp();
    
    let e_val = p_val * m_d;

    e_val as f32
}