# from __future__ import annotations

import re
import os

import itertools
import copy
from enum import IntEnum
from typing import Dict, List, Optional
from pathlib import Path
from PIL import Image
import fitz
from fitz import Rect
from llama_index.readers.base import BaseReader
from llama_index.schema import Document

META_KEY_H = "title_h"
META_KEY_FILE_NAME = "file_name"
IS_DEBUG = False


class ParaTitle(IntEnum):
    TITLE_H1 = (1,)
    TITLE_H2 = (2,)
    TITLE_H3 = (3,)
    TITLE_BODY = 0


def merge_rect(rect1: Rect, rect2: Rect):
    rect1.normalize()
    rect2.normalize()
    merged_x0 = min([rect1.x0, rect2.x0])
    merged_y0 = min([rect1.y0, rect2.y0])
    merged_x1 = max([rect1.x1, rect2.x1])
    merged_y1 = max([rect1.y1, rect2.y1])
    return Rect(merged_x0, merged_y0, merged_x1, merged_y1)


class RenderLine:
    def __init__(self, data_list: list) -> None:
        self.rect = Rect(data_list[0], data_list[1], data_list[2], data_list[3])
        self.rect.normalize()
        self.content = data_list[4]
        self.page_index = 0

    @staticmethod
    def get_list_rect(line_list: List["RenderLine"]) -> Rect:
        if len(line_list) == 0:
            return None

        _merge_rect: Rect = line_list[0].rect
        for line in line_list[1:]:
            _merge_rect = merge_rect(_merge_rect, line.rect)
        return _merge_rect

    @staticmethod
    def get_list_content(line_list: List["RenderLine"]) -> str:
        if len(line_list) == 0:
            return ""

        _merge_content: str = line_list[0].content
        for line in line_list[1:]:
            _merge_content = _merge_content + line.content
        return _merge_content

    @staticmethod
    def create_from_list(in_list: list):
        out_list = []
        for obj in in_list:
            if isinstance(obj, list):
                it_list = [RenderLine(it) for it in obj]
                merge_line = copy.deepcopy(it_list[0])
                merge_line.rect = RenderLine.get_list_rect(it_list)
                merge_line.content = RenderLine.get_list_content(it_list)
                out_list.append(merge_line)
            else:
                out_list.append(RenderLine(obj))

        out_list = [
            line for line in out_list if line.rect.width > 0 and line.rect.height > 0
        ]
        out_list = sorted(out_list, key=lambda x: x.rect.y0)
        return out_list


class RenderPara:
    def __init__(self, line_list: List[RenderLine], index: int) -> None:
        self.line_list = line_list
        for line in line_list:
            line.content.strip()

        self.h_title: int = ParaTitle.TITLE_BODY.value
        self.key_title: str = ""
        self.index: int = index

    def get_content(self):
        return RenderLine.get_list_content(self.line_list)

    def get_line_count(self):
        return len(self.line_list)


class RenderPageManager:
    PAGE_NUMBER_PATTERNS = ["—[0-9]+—", "第[0-9]+页"]

    def __init__(self, filename=None, stream=None) -> None:
        self.pdf = fitz.open(filename=filename, stream=stream)
        self.page_list = self.create_page_list()

        # 设置line的page index
        for index, page in enumerate(self.page_list):
            for line in page:
                line.page_index = index

        self._page_number_res = [
            re.compile(pattern) for pattern in RenderPageManager.PAGE_NUMBER_PATTERNS
        ]
        self._remove_page_number()

        self.page_rect: Rect = self.get_page_rect()
        self.para_list: List[RenderPara] = self.get_para_list()

        self._extract_title()

        if IS_DEBUG:
            dir_path = "debug"
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            self._save_middle_file(f"{dir_path}/line.txt", f"{dir_path}/para.txt")
            for index in range(len(self.page_list)):
                self.draw_page_index(index, dir_path)

    def _is_match_page_number(self, line_list: List[RenderLine]):
        for line in line_list:
            for page_number_re in self._page_number_res:
                if page_number_re.match(line.content):
                    return True
        return False

    def _remove_page_number(self):
        first_line_list = [it[0] for it in self.page_list]
        if self._is_match_page_number(first_line_list):
            self.page_list = [it[1:] for it in self.page_list]

        last_line_list = [it[-1] for it in self.page_list]
        if self._is_match_page_number(last_line_list):
            self.page_list = [it[:-1] for it in self.page_list]

    def get_page_rect(self) -> Rect:
        page_list = self.page_list if len(self.page_list) < 3 else self.page_list[:3]
        line_list = [line for lines in page_list for line in lines]
        return RenderLine.get_list_rect(line_list)

    def _is_end_line(self, line: RenderLine):
        return self.page_rect.x1 - line.rect.x1 < line.rect.height

    def _is_begin_line(self, line: RenderLine, is_first_line: bool):
        if is_first_line:
            return line.rect.x0 - self.page_rect.x0 < 2 * line.rect.height
        else:
            return line.rect.x0 - self.page_rect.x0 < line.rect.height

    def _is_curt_line(self, line: RenderLine):
        if re.match(".*[\)\]\.。？?!！]+$", line.content):
            return False
        return True

    def _is_same_size_line(self, line1: RenderLine, line2: RenderLine):
        return (
            abs(line1.rect.height - line2.rect.height)
            / max(line1.rect.height, line2.rect.height)
            < 0.1
        )

    def _save_middle_file(
        self, line_path: str = "line.txt", para_path: str = "para.txt"
    ):
        with open(line_path, "w") as out_file:
            for para in self.para_list:
                for line in para.line_list:
                    out_file.write("##" + line.content + "\n")

        with open(para_path, "w") as out_file:
            for para in self.para_list:
                out_file.write(para.get_content() + "\n")

    def get_para_list(self):
        line_list = [line for page in self.page_list for line in page]
        out_para = []
        begin_index = 0
        for i in range(len(line_list) - 1):
            line = line_list[i]
            next_line = line_list[i + 1]
            if (
                not self._is_end_line(line)
                or not self._is_curt_line(line)
                or not self._is_begin_line(line, begin_index == i)
                or not self._is_begin_line(next_line, False)
                or not self._is_same_size_line(line, next_line)
            ):
                para_list = (
                    [line_list[begin_index]]
                    if begin_index == i
                    else line_list[begin_index : i + 1]
                )
                out_para.append(para_list)
                begin_index = i + 1
            else:
                continue

        out_para.append(line_list[begin_index:])
        return [RenderPara(para, index) for (index, para) in enumerate(out_para)]

    def create_page_list(self) -> List[List[RenderLine]]:
        page_list = []
        for page in self.pdf:  # iterate the document pages
            text_list = page.get_textpage().extractWORDS()

            line_list = []
            for (block_no, line_no), group in itertools.groupby(
                text_list, lambda x: (x[5], x[6])
            ):
                group = list(group)
                if len(group) == 1:
                    line_list.append(group[0])
                else:
                    line_list.append(group)

            page_list.append(RenderLine.create_from_list(line_list))

        return page_list

    def draw_page_index(self, page_index: int, dir_path: str):
        def _get_para_rect(para: RenderPara, page_index: int) -> Rect:
            line_list = list(
                filter(lambda x: x.page_index == page_index, para.line_list)
            )
            return RenderLine.get_list_rect(line_list)

        page = self.pdf[page_index]
        para_list = list(
            filter(
                lambda para: para.line_list[0].page_index == page_index, self.para_list
            )
        )
        rect_list = [_get_para_rect(para, page_index) for para in para_list]
        for curt_rect in rect_list:
            page.draw_rect(curt_rect)

        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        img_path = os.path.join(dir_path, f"page_{page_index}.png")
        img.save(img_path)

    patterns = [
        r"^第[一二三四五六七八九十百千万零壹贰叁肆伍陆柒捌玖拾佰仟0-9]+([章节卷条回]+)",
    ]

    def _extract_title(self):
        title_list = [it for it in self.para_list if it.get_line_count() == 1]

        for title in title_list:
            title_str = re.sub("\s+", "", title.get_content())
            for pattern in RenderPageManager.patterns:
                curt_match = re.match(pattern, title_str)
                if curt_match:
                    title.key_title = curt_match.groups()[0]

        h_title = 1
        title_dict = {}
        title_list = [title for title in title_list if len(title.key_title) > 0]
        for title in title_list:
            if title.key_title in title_dict:
                title.h_title = title_dict[title.key_title]
            elif h_title <= 3:
                title.h_title = h_title
                title_dict[title.key_title] = h_title
                h_title += 1
            else:
                title.key_title = ""


class FormatPdfReader(BaseReader):
    """PDF parser."""

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        pageManager = RenderPageManager(filename=file)

        docs = []
        for para in pageManager.para_list:
            metadata = {
                "page_label": para.line_list[0].page_index,
                META_KEY_H: para.h_title,
                META_KEY_FILE_NAME: file.name,
            }

            if extra_info is not None:
                metadata.update(extra_info)

            # "file_path"
            docs.append(Document(text=para.get_content(), metadata=metadata, excluded_llm_metadata_keys=["page_label", META_KEY_H, META_KEY_FILE_NAME]))
        return docs
