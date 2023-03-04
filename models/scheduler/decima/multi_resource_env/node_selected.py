class NodeSelected(object):
    def __init__(self, num_exec_groups):
        self.num_exec_groups = num_exec_groups
        self.node_selected = \
            [set() for _ in range(self.num_exec_groups)]

    def __getitem__(self, exec_group_idx):
        return self.node_selected[exec_group_idx]

    def clear(self):
        for i in range(self.num_exec_groups):
            self.node_selected[i].clear()
