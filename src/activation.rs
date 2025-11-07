
pub fn linear(x: f64) -> f64 {
	x
}

pub fn step(x: f64) -> f64 {
	if x>0.0 {1.0} else {0.0}
}

pub fn sigmoid(x: f64) -> f64 {
	(1.0 + (-x).exp()).recip()
}

pub fn hypertan(x: f64) -> f64 {
	let e2w = (x * 2.0).exp();
	(e2w - 1.0) / (e2w + 1.0)
}

pub fn si_lu(x: f64) -> f64 {
	x / (1.0 + (-x).exp())
}

pub fn re_lu(x: f64) -> f64 {
	x.max(0.0)
}
