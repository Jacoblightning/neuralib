use fastnum::{dec64, D64};

pub fn linear(x: D64) -> D64 {
	x
}

pub fn step(x: D64) -> D64 {
	if x>dec64!(0) {dec64!(1)} else {dec64!(0)}
}

pub fn sigmoid(x: D64) -> D64 {
	dec64!(1) / (dec64!(1) + x.neg().exp())
}

pub fn hypertan(x: D64) -> D64 {
	let e2w: D64 = (x * dec64!(2)).exp();
	(e2w - dec64!(1)) / (e2w + dec64!(1))
}

pub fn si_lu(x: D64) -> D64 {
	x / (dec64!(1) + x.neg().exp())
}

pub fn re_lu(x: D64) -> D64 {
	x.max(dec64!(0))
}
