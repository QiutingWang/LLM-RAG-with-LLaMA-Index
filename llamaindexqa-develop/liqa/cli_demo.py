import os
import sys
from colorama import Fore, Style

CURT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURT_PATH))


def clear_screen(config):
    os.system("clear")
    print(
        Fore.YELLOW
        + Style.BRIGHT
        + "欢迎使用本地知识库beta版，输入进行对话，clear 清空重来，CTRL+C 中断，exit 结束。"
    )
    print(Fore.YELLOW + Style.BRIGHT + "请选择数据源：1.城投; 2.中行证券;")
    prompt = input(Fore.GREEN + Style.BRIGHT + "输入：" + Style.NORMAL)
    curt_path = os.path.join(
        CURT_PATH, "database", {"1": "城投", "2": "中行证券"}.get(prompt)
    )

    config.pdf_path = curt_path
    assembler = PiplineRun(config)
    return assembler


def main(if_openai: bool = False):
    curt_config = OmegaConf.load(os.path.join(CURT_PATH, "config.yaml"))
    utils.deal_obj_inherited_and_path(curt_config)

    assembler = clear_screen(curt_config.model_default)
    while True:
        query = input(Fore.GREEN + Style.BRIGHT + "请输入问题：" + Style.NORMAL)
        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            assembler = clear_screen(curt_config.model_default)
            continue

        answer = assembler.query_content(query)
        query = print(Fore.YELLOW + f"回答：\n{answer}" + Style.NORMAL)

    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
