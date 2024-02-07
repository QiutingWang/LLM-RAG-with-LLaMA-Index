"""Simple node parser."""

import itertools
from typing import List, Optional, Sequence, Any

from llama_index.bridge.pydantic import Field
from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.node_parser.interface import NodeParser
from llama_index.schema import TextNode, BaseNode, Document, NodeRelationship

from liqa.load.format_pdf_reader import ParaTitle, META_KEY_H
from liqa.load import load_util


class FormatNodeParser(NodeParser):
    """Simple node parser.

    Splits a document into Nodes using a TextSplitter.

    Args:
        text_splitter (Optional[TextSplitter]): text splitter
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships

    """
    chunk_size: Optional[int] = Field(
        default=512, description="Whether or not to consider metadata when splitting."
    )    
    include_metadata: bool = Field(
        default=True, description="Whether or not to consider metadata when splitting."
    )
    include_prev_next_rel: bool = Field(
        default=True, description="Include prev/next node relationships."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )
    parent_has_child_content: bool = Field(
        default=True, description="parent has childs content."
    )

    @classmethod
    def from_defaults(
        cls,
        chunk_size: Optional[int] = 512,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        parent_has_child_content: bool = True,
    ) -> "FormatNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            chunk_size=chunk_size,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            parent_has_child_content=parent_has_child_content,
        )

    @classmethod
    def class_name(cls) -> str:
        return "FormatNodeParser"

    def _merge_split_nodes(self, nodes: List[BaseNode]):
        """
        Merges split nodes into a single node.

        Args:
            nodes (List[BaseNode]): List of split nodes.

        Returns:
            List[BaseNode]: List of merged nodes.
        """
        if len(nodes) == 0:
            return nodes

        def _merge_node(nodes: List[BaseNode]) -> List[BaseNode]:
            merged_nodes = []
            curt_node = nodes[0]
            for node in nodes[1:]:
                if len(node.get_content()) + len(curt_node.get_content()) <= self.chunk_size:
                    curt_node.set_content(
                        f"{curt_node.get_content()}\n{node.get_content()}"
                    )
                    node.set_content("")
                else:
                    merged_nodes.append(curt_node)
                    curt_node = node

            merged_nodes.append(curt_node)
            return merged_nodes

        patterns = [
            "^\s*(\(\d+\)|\d+\.\d+|\d+)",
            "^\s*（[[一二三四五六七八九十百千万零壹贰叁肆伍陆柒捌玖拾佰仟0-9]+）",
        ]

        merged_nodes = []
        number_if_list = [
            load_util.get_text_math_patterns(patterns, node.get_content()) is not None
            for node in nodes
        ]

        # 合并结点，如果是列表，列表前一段一般是提示性的内容，需要和列表合并在一起
        for index in range(len(number_if_list) - 1):
            if not number_if_list[index] and number_if_list[index + 1]:
                number_if_list[index] = True

        for _, group in itertools.groupby(zip(nodes, number_if_list), lambda x: x[1]):
            group = list(group)
            merged_nodes.extend(_merge_node([it[0] for it in group]))
        return merged_nodes

    def _create_node_tree(self, nodes: List[BaseNode], level: int = ParaTitle.TITLE_H1):
        if level > ParaTitle.TITLE_H3 or len(nodes) == 0:
            return self._merge_split_nodes(nodes)

        if (
            len(
                list(
                    filter(
                        lambda node: node.metadata[META_KEY_H] != ParaTitle.TITLE_BODY,
                        nodes,
                    )
                )
            )
            == 0
        ):
            return self._merge_split_nodes(nodes)

        def _split(nodes: List[BaseNode], level):
            begin_index = 0
            out_list = []
            for index, node in enumerate(nodes):
                if int(node.metadata[META_KEY_H]) == level:
                    out_list.append(nodes[begin_index:index])
                    begin_index = index
            out_list.append(nodes[begin_index:])

            return out_list[0], [] if len(out_list) == 1 else [
                (it[0], it[1:]) for it in out_list[1:]
            ]

        first_list, pair_list = _split(nodes, level)
        nodes = []
        nodes.extend(self._create_node_tree(first_list, level + 1))
        for parent, child_list in pair_list:
            child_nodes = self._create_node_tree(child_list, level + 1)
            for child_node in child_nodes:
                load_util.add_parent_child_relationship(parent, child_node)

            if self.parent_has_child_content:
                parent.set_content(
                    f"{parent.get_content()}\n{load_util.get_child_content(child_nodes)}"
                )

            nodes.append(parent)

        if self.include_prev_next_rel:
            for i, node in enumerate(nodes):
                if i > 0:
                    node.relationships[NodeRelationship.PREVIOUS] = nodes[
                        i - 1
                    ].as_related_node_info()
                if i < len(nodes) - 1:
                    node.relationships[NodeRelationship.NEXT] = nodes[
                        i + 1
                    ].as_related_node_info()
        return nodes

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = nodes
        excluded_llm_metadata_keys = nodes[0].excluded_llm_metadata_keys

        for _, group in itertools.groupby(
            all_nodes, key=lambda x: x.metadata["file_name"]
        ):
            self._create_node_tree(list(group))
        all_nodes = [node for node in all_nodes if node.get_content() != ""]
        
        for node in all_nodes:
            node.excluded_llm_metadata_keys = excluded_llm_metadata_keys
        return all_nodes
