# Utilities for importing torch models into MLIR keeping high-level ops
# atomic (in particular, batch norm is not decomposed).

import math

import torch
from torch_mlir import fx


def _rewrite_graph(graph):
    """Rewrites the exported graph into the subset the torch-mlir frozen
    import + NN conversion can handle:

      aten.relu_.default          -> aten.relu.default
      aten.add_.Tensor            -> aten.add.Tensor
      aten.silu_.default          -> aten.silu.default
      aten.chunk.default          -> aten.slice.Tensor (per getitem use)
      aten.split_with_sizes.default -> aten.slice.Tensor (per getitem use)

    In-place variants are kept pure because the frozen import emits them with
    `!torch.vtensor` operands, which their ODS definitions (operands typed
    `!torch.tensor`) reject at verification time. `chunk`/`split_with_sizes`
    produce a list that the NN dialect has no lowering for, so each extracted
    element is rewritten into a `slice.Tensor` with its fake-tensor metadata
    filled in. Batch norm needs no rewriting: the importer maps
    `_native_batch_norm_legit_no_training` to the canonical 9-operand
    `aten.batch_norm` itself (with `decomposition_table={}` preventing
    decomposition).
    """
    for node in list(graph.nodes):
        target = node.target
        if target == torch.ops.aten.relu_.default:
            node.target = torch.ops.aten.relu.default
        elif target == torch.ops.aten.add_.Tensor:
            node.target = torch.ops.aten.add.Tensor
        elif target == torch.ops.aten.silu_.default:
            node.target = torch.ops.aten.silu.default
        elif target == torch.ops.aten.chunk.default:
            x, n, dim = node.args
            dim = dim if dim >= 0 else dim + x.meta["val"].ndim
            dim_size = x.meta["val"].shape[dim]
            chunk_size = math.ceil(dim_size / n)
            for getitem in list(node.users):
                idx = getitem.args[1]
                start = idx * chunk_size
                end = min(start + chunk_size, dim_size)
                with graph.inserting_before(getitem):
                    sl = graph.call_function(
                        torch.ops.aten.slice.Tensor, (x, dim, start, end, 1)
                    )
                sl.meta["val"] = x.meta["val"][
                    (slice(None),) * dim + (slice(start, end),)
                ]
                getitem.replace_all_uses_with(sl)
                graph.erase_node(getitem)
            graph.erase_node(node)
        elif target == torch.ops.aten.split_with_sizes.default:
            x, sizes, dim = node.args
            dim = dim if dim >= 0 else dim + x.meta["val"].ndim
            sizes = [int(s) for s in sizes]
            for getitem in list(node.users):
                idx = getitem.args[1]
                start = sum(sizes[:idx])
                end = start + sizes[idx]
                with graph.inserting_before(getitem):
                    sl = graph.call_function(
                        torch.ops.aten.slice.Tensor, (x, dim, start, end, 1)
                    )
                sl.meta["val"] = x.meta["val"][
                    (slice(None),) * dim + (slice(start, end),)
                ]
                getitem.replace_all_uses_with(sl)
                graph.erase_node(getitem)
            graph.erase_node(node)


def export_for_backend(model, example_input):
    """Exports `model` with torch.export and rewrites the FX graph so that
    in-place ops are kept as pure high-level ops and list-producing split ops
    are replaced by slices (see `_rewrite_graph`).
    """
    prog = torch.export.export(model, (example_input,))
    graph = prog.graph_module.graph
    _rewrite_graph(graph)
    graph.lint()
    return prog


def export_and_import_for_backend(model, example_input):
    """Imports `model` into a torch-mlir module at the backend contract level
    with atomic batch norm (no decomposition).
    """
    prog = export_for_backend(model, example_input)
    return fx.export_and_import(prog, decomposition_table={})
