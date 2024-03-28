use std::{fs::File, io::Write};

use strum::IntoEnumIterator;
use tera::Context;

use crate::{Generate, KernelElement, KernelRenderer, WgslDType};

#[derive(Debug, Clone, strum_macros::EnumIter)]
pub enum UnaryOp {
    Gelu,
    Tanh,
    Exp,
    Log,
    Sin,
    Cos,
    Abs,
    Sqrt,
    Relu,
    Floor,
    Ceil,
}

impl std::fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            UnaryOp::Gelu => "gelu",
            UnaryOp::Tanh => "tanh",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Abs => "abs",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Relu => "relu",
            UnaryOp::Floor => "floor",
            UnaryOp::Ceil => "ceil",
        };
        write!(f, "{}", s)
    }
}

impl Generate for UnaryOp {
    fn generate(renderer: &mut KernelRenderer) -> anyhow::Result<()> {
        for inplace in [true, false].iter() {
            for func in UnaryOp::iter() {
                for ke in KernelElement::iter() {
                    let path = renderer.templates_path.join("unary.wgsl");
                    renderer.tera.add_template_file(path, Some("unary"))?;

                    let mut context = Context::new();
                    let tera_func = match func {
                        UnaryOp::Tanh => String::from("safe_tanh"),
                        _ => func.to_string(),
                    };
                    context.insert("inplace", inplace);
                    context.insert("func", &tera_func);
                    context.insert("elem", &ke.as_wgsl(WgslDType::F32));
                    context.insert("elem_size", &ke.as_size());
                    let rendered = renderer.tera.render("unary", &context)?;

                    let kernel_fname = if *inplace {
                        format!("{}_inplace_{}.wgsl", func, ke)
                    } else {
                        format!("{}_{}.wgsl", func, ke)
                    };
                    let mut file = File::create(renderer.dest_path.join(kernel_fname))?;
                    file.write_all(rendered.as_bytes())?;
                }
            }
        }
        Ok(())
    }
}
