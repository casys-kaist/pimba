import heapq
import itertools
import math
from collections import defaultdict
from typing import Tuple


def get_min_key(d):
    min_value = min(d.values())
    for key, value in d.items():
        if value == min_value:
            return key


class Trace:
    atom_size = 32
    page_size = 1024
    hierarchy = (8, 2, 4, 4)  # channels, pseudochannels, bankgroups, banks

    def __init__(
        self,
        op_type: str,
        num_op: int,
        state_size: Tuple[int, int],
        hw: str,
        num_channels: int,
        elem_width: int,
        use_chunk_group: bool,
        use_command_scheduling: bool,
    ):
        self.op_type = op_type
        self.num_op = num_op
        self.state_size = state_size
        self.hw = hw
        self.elem_width = elem_width
        self.use_chunk_group = use_chunk_group
        self.use_command_scheduling = use_command_scheduling
        self.hierarchy = (num_channels, *self.hierarchy[1:])

    def name(self):
        return f"op_type_{self.op_type}-hw_{self.hw}-num_op_{self.num_op}-state-{self.state_size[0]}x{self.state_size[1]}-channels_{self.hierarchy[0]}-elem_width_{self.elem_width}-chunk_group_{self.use_chunk_group}-command_scheduling_{self.use_command_scheduling}.trace"

    def generate(self):
        chunk_rows = self.atom_size // self.elem_width
        chunk_cols = self.page_size // self.atom_size
        v_count = chunk_cols * self.elem_width // self.atom_size

        num_chunks_r = math.ceil(self.state_size[0] / chunk_rows) * self.num_op
        num_chunks_c = math.ceil(self.state_size[1] / chunk_cols)
        num_chunks = num_chunks_r * num_chunks_c

        block_row = defaultdict(lambda: None)
        block_col = {i: 0 for i in range(num_chunks_r)}
        # Initialize min-heap with (block_col_value, block_row_index)
        min_heap = [(0, i) for i in range(num_chunks_r)]
        heapq.heapify(min_heap)

        cur_idx = 0
        res_map = defaultdict(list)
        while True:
            for ch in range(self.hierarchy[0]):  # channel
                for pch in range(self.hierarchy[1]):  # pseudochannel
                    # act & reg
                    temp = []
                    acc_reg_write_count = 0
                    for bg in range(self.hierarchy[2]):  # bankgroup
                        reg_write_count = 0
                        for bank in range(self.hierarchy[3]):  # bank
                            cur_block_row = block_row[(ch, pch, bg, bank)]
                            if (
                                cur_block_row is None
                                or block_col[cur_block_row] == num_chunks_c
                            ):
                                # Get the block_row with minimal block_col value
                                while True:
                                    if not min_heap:
                                        raise Exception("Ran out of block rows")
                                    min_col_value, min_block_row = heapq.heappop(
                                        min_heap
                                    )
                                    if (
                                        block_col[min_block_row] == min_col_value
                                        and block_col[min_block_row] < num_chunks_c
                                    ):
                                        cur_block_row = min_block_row
                                        break
                                block_row[(ch, pch, bg, bank)] = cur_block_row
                                block_col[cur_block_row] += 1
                                heapq.heappush(
                                    min_heap, (block_col[cur_block_row], cur_block_row)
                                )
                                if self.op_type == "SU":
                                    reg_write_count += 3 + v_count
                                else:
                                    reg_write_count += 1
                            else:
                                block_col[cur_block_row] += 1
                                heapq.heappush(
                                    min_heap, (block_col[cur_block_row], cur_block_row)
                                )
                                if self.op_type == "SU":
                                    if self.use_chunk_group:
                                        reg_write_count += v_count
                                    else:
                                        reg_write_count += 3 + v_count
                                else:
                                    reg_write_count += 1
                        if self.use_command_scheduling:
                            temp.append(f"act4_with_reg_{reg_write_count} {ch} {pch}\n")
                        else:
                            temp.append(f"act4 {ch} {pch}\n")
                            temp.extend([f"reg_write {ch} {pch}\n"] * reg_write_count)
                        acc_reg_write_count += reg_write_count

                    # For simulator behavior
                    if self.op_type == "SU":
                        if self.hw == "Pimba" or self.hw == "Pipelined":
                            comps = chunk_cols * 2
                        elif self.hw == "HBM-PIM":
                            comps = chunk_cols * 8
                        elif self.hw == "Time-multiplexed":
                            comps = chunk_cols * 4
                        else:
                            raise NotImplementedError
                    else:
                        comps = chunk_cols * 2

                    res_map[(ch, pch)].append(
                        f"refresh_barrier {ch} {pch} {1 if self.use_command_scheduling else 0} {acc_reg_write_count} {v_count} {comps}\n"
                    )
                    res_map[(ch, pch)].extend(temp)

                    # comp
                    res_map[(ch, pch)].extend([f"comp {ch} {pch}\n"] * comps)

                    # precharge & result
                    if self.op_type == "ATTEND":
                        res_count = 8
                    else:
                        res_count = v_count * 8
                    if self.use_command_scheduling:
                        res_map[(ch, pch)].append(
                            f"precharge_with_result_{res_count} {ch} {pch}\n"
                        )
                    else:
                        res_map[(ch, pch)].append(f"precharge {ch} {pch}\n")
                        res_map[(ch, pch)].extend([f"result {ch} {pch}\n"] * res_count)

                    cur_idx += self.hierarchy[2] * self.hierarchy[3]
                    if cur_idx >= num_chunks:
                        t = list(itertools.chain(*zip(*res_map.values())))
                        return "".join(t)
