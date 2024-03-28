#![cfg(feature = "plotting")]
use crate::{Tensor, TensorId};
use derive_new::new;
use std::{
    borrow::Cow,
    collections::HashMap,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};
use tempfile::NamedTempFile;

type Nd = usize;

#[derive(Clone, Debug, new)]
struct PlotEdge {
    id: usize,
    label: Cow<'static, str>,
    from: usize,
    to: usize,
    attributes: Option<HashMap<&'static str, &'static str>>,
}

impl PlotEdge {
    fn add_attribute(&mut self, attribute: (&'static str, &'static str)) {
        if let Some(attr_map) = &mut self.attributes {
            attr_map.insert(attribute.0, attribute.1);
        } else {
            self.attributes = Some(HashMap::from([attribute]));
        }
    }

    fn style_as_inplace(&mut self) {
        self.add_attribute(("color", "red"));
    }

    fn style_as_normal(&mut self) {
        self.add_attribute(("color", "black"));
    }
}

#[derive(Clone, Debug, new)]
struct PlotNode {
    plot_id: usize,
    tensor_id: TensorId,
    op_type: Cow<'static, str>,
    attributes: Option<HashMap<&'static str, &'static str>>,
}

impl PlotNode {
    fn add_attribute(&mut self, attribute: (&'static str, &'static str)) {
        if let Some(attr_map) = &mut self.attributes {
            attr_map.insert(attribute.0, attribute.1);
        } else {
            self.attributes = Some(HashMap::from([attribute]));
        }
    }

    fn style_as_op(&mut self) {
        self.add_attribute(("fillcolor", "white"));
        self.add_attribute(("shape", "box"));
    }

    fn style_as_output(&mut self) {
        self.add_attribute(("fillcolor", "lightgray"));
        self.add_attribute(("shape", "ellipse"));
    }

    fn style_as_const(&mut self) {
        self.add_attribute(("fillcolor", "lightgreen"));
        self.add_attribute(("shape", "ellipse"));
    }

    fn style_as_inplace(&mut self) {
        self.add_attribute(("fillcolor", "white"));
        self.add_attribute(("color", "red"));
    }
}

#[derive(Default, Debug)]
struct RenderableGraph {
    current_node_id: usize,
    current_edge_id: usize,
    nodes: Vec<PlotNode>,
    edges: Vec<PlotEdge>,
}

impl RenderableGraph {
    fn new() -> Self {
        RenderableGraph {
            current_node_id: 0,
            current_edge_id: 0,
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    fn create_node(&mut self, actual_id: TensorId, op_type: Cow<'static, str>) -> &mut PlotNode {
        let n = PlotNode::new(self.current_node_id, actual_id, op_type, None);
        self.nodes.push(n);
        self.current_node_id += 1;
        &mut self.nodes[self.current_node_id - 1]
    }

    fn create_edge(&mut self, label: Cow<'static, str>, from: usize, to: usize) -> &mut PlotEdge {
        let e = PlotEdge::new(self.current_edge_id, label, from, to, None);
        self.edges.push(e);
        self.current_edge_id += 1;
        &mut self.edges[self.current_edge_id - 1]
    }

    fn build_graph(leaf: &Tensor) -> anyhow::Result<RenderableGraph> {
        log::warn!("Rendering plot");
        let mut g = RenderableGraph::new();

        let mut graph_index_map = HashMap::new();
        let execution_order = leaf.execution_order();
        for t in execution_order.iter() {
            let renderable_node = g.create_node(t.id(), Cow::Owned(t.op().name()));
            let can_inplace = t.op().supports_inplace() && Arc::strong_count(&t.inner) == 1;
            match t.op() {
                crate::LazyOp::Const => renderable_node.style_as_const(),
                _ => renderable_node.style_as_op(),
            }
            if can_inplace {
                renderable_node.style_as_inplace()
            }

            let node_graph_id = renderable_node.plot_id;
            graph_index_map.insert(t.id(), renderable_node.plot_id);
            t.op().srcs().iter().for_each(|src_t| {
                if let Some(src_id) = graph_index_map.get(&src_t.id()) {
                    let e = g.create_edge(Cow::Owned(src_t.plot_fmt()), *src_id, node_graph_id);
                } else {
                    panic!("Source tensor not found in graph index map");
                }
            });
        }

        g.create_node(leaf.id(), Cow::Owned(leaf.op().name()))
            .style_as_output();

        Ok(g)
    }

    fn plot_to_file(self, fname: impl AsRef<Path>) -> anyhow::Result<()> {
        let mut d = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d.push(fname.as_ref());
        let mut f = NamedTempFile::new().expect("Failed to create temp file.");
        render_to(&mut f, self)?;
        Command::new("dot")
            .arg("-Tsvg")
            .arg(f.path())
            .arg("-o")
            .arg(d)
            .output()?;
        Ok(())
    }
}

fn render_to<W: Write>(output: &mut W, graph: RenderableGraph) -> anyhow::Result<()> {
    Ok(dot3::render(&graph, output)?)
}

impl<'a> dot3::Labeller<'a, Nd, PlotEdge> for RenderableGraph {
    fn graph_attrs(&'a self) -> HashMap<&str, &str> {
        HashMap::from([("ordering", "in")])
    }

    fn graph_id(&'a self) -> dot3::Id<'a> {
        dot3::Id::new("test").unwrap()
    }

    fn node_id(&'a self, n: &Nd) -> dot3::Id<'a> {
        dot3::Id::new(format!("N{:?}", self.nodes[*n].tensor_id)).unwrap()
    }

    fn node_label<'b>(&'b self, n: &Nd) -> dot3::LabelText<'b> {
        dot3::LabelText::LabelStr(self.nodes[*n].op_type.clone())
    }

    fn edge_label<'b>(&'b self, e: &PlotEdge) -> dot3::LabelText<'b> {
        dot3::LabelText::LabelStr(e.label.clone())
    }

    fn node_color(&'a self, _node: &Nd) -> Option<dot3::LabelText<'a>> {
        Some(dot3::LabelText::LabelStr("black".into()))
    }

    fn node_style(&'a self, _n: &Nd) -> dot3::Style {
        dot3::Style::Filled
    }

    fn node_attrs(&'a self, n: &Nd) -> HashMap<&str, &str> {
        self.nodes[*n]
            .attributes
            .clone()
            .unwrap_or_else(|| HashMap::from([("fillcolor", "white")]))
    }

    fn edge_attrs(&'a self, e: &PlotEdge) -> HashMap<&str, &str> {
        self.edges[e.id]
            .attributes
            .clone()
            .unwrap_or_else(|| HashMap::from([("color", "black")]))
    }
}

impl<'a> dot3::GraphWalk<'a, Nd, PlotEdge> for RenderableGraph {
    fn nodes(&self) -> dot3::Nodes<'a, Nd> {
        (0..self.nodes.len()).collect()
    }
    fn edges(&'a self) -> dot3::Edges<'a, PlotEdge> {
        self.edges.clone().into_iter().collect()
    }
    fn source(&self, e: &PlotEdge) -> Nd {
        e.from
    }
    fn target(&self, e: &PlotEdge) -> Nd {
        e.to
    }
}

pub fn render_to_file(t: &Tensor, fname: impl AsRef<Path>) -> anyhow::Result<()> {
    RenderableGraph::plot_to_file(RenderableGraph::build_graph(t)?, fname)
}
