use std::cmp::Ordering;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SizeDistPriorityInfo {
    pub position: usize,
    pub tensor_size: usize,
    pub dist: Vec<usize>,
    pub best_dist: usize,
    pub best_object: usize,
    pub tensor_usage_id: usize, //INDEX
}

impl Default for SizeDistPriorityInfo {
    fn default() -> Self {
        Self {
            position: 0,
            tensor_size: 0,
            dist: Vec::new(),
            best_dist: usize::MAX,
            best_object: usize::MAX,
            tensor_usage_id: 0,
        }
    }
}

impl PartialOrd for SizeDistPriorityInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl SizeDistPriorityInfo {
    pub fn recalc_best_dist(&mut self) {
        self.best_dist = std::usize::MAX;
        for (obj_id, dist) in self.dist.iter().enumerate() {
            if *dist < self.best_dist {
                self.best_dist = *dist;
                self.best_object = obj_id;
            }
        }
    }
}

impl Ord for SizeDistPriorityInfo {
    fn cmp(&self, other: &Self) -> Ordering {
        let condition = self.position < other.position
            || (self.position == other.position
                && (self.best_dist < other.best_dist
                    || (self.best_dist == other.best_dist
                        && self.tensor_size > other.tensor_size)));
        if condition {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}
