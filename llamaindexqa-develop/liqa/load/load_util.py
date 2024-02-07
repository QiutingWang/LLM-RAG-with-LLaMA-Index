from typing import Union, List
import re
from llama_index.schema import BaseNode, Document, NodeRelationship, RelatedNodeInfo
from liqa.load.format_pdf_reader import ParaTitle, META_KEY_H, META_KEY_FILE_NAME


def add_parent_child_relationship(parent_node: BaseNode, child_node: BaseNode) -> None:
    """Add parent/child relationship between nodes."""
    child_list = parent_node.relationships.get(NodeRelationship.CHILD, [])
    child_list.append(child_node.as_related_node_info())
    parent_node.relationships[NodeRelationship.CHILD] = child_list

    child_node.relationships[
        NodeRelationship.PARENT
    ] = parent_node.as_related_node_info()


def get_child_content(nodes: List[BaseNode]) -> str:
    if len(nodes) == 0:
        return ""

    content = nodes[0].get_content()
    for node in nodes[1:]:
        content = f"{content}\n{node.get_content()}"

    return content


def get_relation_nodes(
    total_nodes: List[BaseNode],
    related_obj: Union[RelatedNodeInfo, List[RelatedNodeInfo]],
):
    def _take(total_nodes: List[BaseNode], related_obj: RelatedNodeInfo):
        for node in total_nodes:
            if node.node_id == related_obj.node_id:
                return node
        assert False, "input node error"

    if isinstance(related_obj, RelatedNodeInfo):
        return _take(total_nodes, related_obj)
    if isinstance(related_obj, List):
        return [_take(total_nodes, node) for node in related_obj]
    assert False, "input node error"


def _make_parent_contain_child_content(total_nodes: List[BaseNode], node: BaseNode):
    if len(node.child_nodes) == 0:
        return

    child_nodes = get_relation_nodes(total_nodes, node.child_nodes)
    for node in child_nodes:
        _make_parent_contain_child_content(total_nodes, node)

    node.set_content(f"{node.get_content()}\n{get_child_content(child_nodes)}")


def make_parent_contain_child_content(total_nodes: List[BaseNode]):
    for node in total_nodes:
        _make_parent_contain_child_content(total_nodes, node)


def nodes_save_file(nodes: List[BaseNode], file_path: str):
    with open(file_path, "w") as out_file:
        for node in nodes:
            out_file.write(
                f"##{node.metadata[META_KEY_FILE_NAME]} {node.get_content()}\n"
            )


def get_nodes(nodes: List[BaseNode], title: ParaTitle) -> List[BaseNode]:
    """Get leaf nodes."""
    return list(filter(lambda node: node.metadata[META_KEY_H] == title, nodes))


def get_text_math_patterns(patterns:List[str], text:str):
    """
    Check if the given text matches any of the given patterns.

    Args:
        patterns (List[str]): A list of regex patterns to match against the text.
        text (str): The text to match against.

    Returns:
        If a match is found, returns the match object. Otherwise, returns None.
    """
    for pattern in patterns:
        curt_match = re.match(pattern, text)
        if curt_match:
            return curt_match
    return None