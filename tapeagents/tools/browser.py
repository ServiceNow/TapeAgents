import os
import re
from time import sleep
from typing import Any, Callable, Literal
from uuid import uuid4

import gymnasium as gym
import markdownify
import numpy as np
import requests
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.env import BrowserEnv
from browsergym.utils.obs import (
    IGNORED_AXTREE_PROPERTIES,
    _process_bid,
    flatten_dom_to_str,
    prune_html,
)
from bs4 import BeautifulSoup
from PIL import Image
from pydantic import Field

from tapeagents.core import Action, Observation, StepMetadata
from tapeagents.tools.base import Multitool
from tapeagents.tools.document_reader import read_document
from tapeagents.tools.simple_browser import PageObservation

NODES_WITH_BID = [
    "button",
    "link",
    "combobox",
    "checkbox",
    "textbox",
    "input",
    "radio",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "LabelText",
    "tab",
]


class GotoPageAction(Action):
    """
    Action that opens the page with the provided URL and returns the first page of its content.
    To read the following pages use scroll_action.
    """

    kind: Literal["goto_page_action"] = "goto_page_action"
    url: str = Field(description="url to go to")


class GoForwardAction(Action):
    """
    Action that goes forward in the browser history
    """

    kind: Literal["go_forward_action"] = "go_forward_action"


class GoBackAction(Action):
    """
    Action that goes back in the browser history
    """

    kind: Literal["go_back_action"] = "go_back_action"


class CloseTabAction(Action):
    """
    Action that closes the browser tab
    """

    kind: Literal["close_tab_action"] = "close_tab_action"


class NewTabAction(Action):
    """
    Action that opens a new browser tab
    """

    kind: Literal["new_tab_action"] = "new_tab_action"


class TabFocusAction(Action):
    """
    Action that focuses the tab with the provided index
    """

    kind: Literal["tab_focus_action"] = "tab_focus_action"
    index: int = Field(description="index of the tab to focus")


class ScrollAction(Action):
    """
    Action that scrolls the page in the provided direction and returns the next page of content.
    """

    kind: Literal["scroll_action"] = "scroll_action"
    direction: str = Field(description="direction to scroll")


class PressAction(Action):
    """
    Action that puts focus on the element with a given BID and presses a combination of keys.
    Accepts the logical key names: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDown, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc.
    Following modification, shortcuts are also supported: Shift, Control, Alt, Meta.
    """

    kind: Literal["press_action"] = "press_action"
    bid: str = Field(description="BID of the input element to focus")
    key_comb: str = Field(description="keys combination to press")


class InputTextAction(Action):
    """
    Action that fills out the input element identified by BID with the provided text
    """

    kind: Literal["input_text_action"] = "input_text_action"
    bid: str = Field(description="BID of the input element to fill")
    text: str = Field(description="text to put into the element")


class HoverAction(Action):
    """
    Action that hovers over the element on the page with the provided BID
    """

    kind: Literal["hover_action"] = "hover_action"
    bid: str = Field(description="BID of the element to hover")


class SelectOptionAction(Action):
    """
    Action that selects option in the dropdown or combobox element with the provided BID.
    ONLY applicable to dropdowns and comboboxes!
    """

    kind: Literal["select_option_action"] = "select_option_action"
    bid: str = Field(description="BID of the dropdown or combobox to select from")
    element_description: str = Field(description="brief description of the dropdown or combobox")
    option: str = Field(description="option to select")


class ClickAction(Action):
    """
    Action that clicks the element on the page with the provided BID
    """

    kind: Literal["click_action"] = "click_action"
    bid: str = Field(description="BID of the element to click")
    button: Literal["left", "middle", "right"] = Field(description="button to click", default="left")
    modifiers: list[Literal["Alt", "Control", "Meta", "Shift"]] = Field(
        description="modifier keys to press", default_factory=list
    )


class Browser(Multitool):
    actions: tuple[type[Action], ...] = (
        ClickAction,
        GotoPageAction,
        GoBackAction,
        GoForwardAction,
        HoverAction,
        InputTextAction,
        PressAction,
        ScrollAction,
        SelectOptionAction,
    )
    observations: tuple[type[Observation], ...] = (PageObservation,)
    tab_actions: list[type[Action]] = [CloseTabAction, NewTabAction, TabFocusAction]
    axtree: bool = True
    viewport_size: int = 64000
    timeout_ms: int = 30000
    headless: bool = True
    save_video: bool = False
    exp_path: str | None = None
    page_load_time_sec: int = 1
    gym_kwargs: dict = {}
    gym_task: str = "browsergym/openended"

    _env: BrowserEnv = None  # type: ignore
    _current_page: str = ""
    _current_viewport: int = 0
    _n_viewports: int = 1
    _action_map: dict[type[Action], Callable] = {}
    _traces_dir: str | None = None
    _record_video_dir: str | None = None
    _screenshots_dir: str | None = None
    _task_id: str = ""

    def model_post_init(self, __context: Any):
        self._current_page = ""
        self._current_viewport = 1
        self._n_viewports = 1
        self._action_map = {
            ClickAction: self.click,
            SelectOptionAction: self.select_option,
            CloseTabAction: self.close_tab,
            InputTextAction: self.input_text,
            GoBackAction: self.go_back,
            GoForwardAction: self.go_forward,
            GotoPageAction: self.goto_page,
            HoverAction: self.hover,
            NewTabAction: self.new_tab,
            PressAction: self.press,
            ScrollAction: self.scroll,
            TabFocusAction: self.tab_focus,
        }
        if self.exp_path:
            assert os.path.isdir(self.exp_path)
            self._traces_dir = os.path.join(self.exp_path, "playwright_traces")
            self._record_video_dir = os.path.join(self.exp_path, "attachments/browser/videos")
            self._screenshots_dir = os.path.join(self.exp_path, "attachments/browser/screenshots")
            os.makedirs(self._traces_dir, exist_ok=True)
            os.makedirs(self._record_video_dir, exist_ok=True)
            os.makedirs(self._screenshots_dir, exist_ok=True)

        self._env = gym.make(
            self.gym_task,
            headless=self.headless,
            record_video_dir=self._record_video_dir if self.save_video else None,
            action_mapping=HighLevelActionSet(demo_mode="default").to_python_code,
            timeout=self.timeout_ms,
            task_kwargs={"start_url": "about:blank"},
            **self.gym_kwargs,
        )  # type: ignore
        while not isinstance(self._env, BrowserEnv):
            self._env = self._env.env
        self._env.reset()
        self._env.context.tracing.start(screenshots=True, snapshots=True)

    def execute_action(self, action: Action) -> PageObservation:
        action_type = type(action)
        if action_type in self._action_map:
            return self._action_map[action_type](action)
        raise ValueError(f"Unknown action: {action_type}")

    def start_task(self, task_id: str, seed: int = 1, **kwargs) -> dict:
        self._task_id = task_id
        self._env = gym.make(
            task_id,
            headless=self.headless,
            record_video_dir=self._record_video_dir if self.save_video else None,
            action_mapping=HighLevelActionSet(demo_mode="default").to_python_code,
            timeout=self.timeout_ms,
            **kwargs,
        )  # type: ignore
        start_obs, info = self._env.reset(seed=seed)
        self._env.context.tracing.start(screenshots=True, snapshots=True)
        self._env.chat.add_message(role="assistant", msg="Running TapeAgent...")
        assert self._env.task is not None
        info = {
            "name": self._env.task.get_task_id(),
            "goal": start_obs["goal"],
            "task_info": info["task_info"],
            "video": os.path.basename(self._env.page.video.path()) if self._env.page.video else "",
            "chat_video": os.path.basename(self._env.chat.page.video.path()) if self._env.chat.page.video else "",
        }
        sleep(self.page_load_time_sec)  # wait for the page to load
        return info

    def close(self):
        assert self._traces_dir is not None
        self._env.context.tracing.stop(path=os.path.join(self._traces_dir, f"{self._task_id}.zip"))
        self._env.close()

    def _screenshot_to_img_file(self, image) -> str:
        if self._screenshots_dir is None:
            return ""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        pic_uid = uuid4().hex
        img_path = os.path.join(self._screenshots_dir, f"{pic_uid}.png")
        image.save(img_path)
        return os.path.relpath(img_path, self._screenshots_dir)

    def run_browser_action(self, action_text: str) -> PageObservation:
        obs_dict, reward, terminated, truncated, info = self._env.step(action_text)
        error = self.format_error(obs_dict["last_action_error"])
        if error:
            content = ""
        elif self.axtree:
            content = flatten_axtree(obs_dict["axtree_object"])
        else:
            html_content = prune_html(flatten_dom_to_str(obs_dict["dom_object"]))
            content = self.html_to_markdown(html_content)

        screen_path = self._screenshot_to_img_file(obs_dict["screenshot"])
        observation = PageObservation(
            text=self.get_viewport(content),
            current_page=self._current_viewport,
            total_pages=self._n_viewports,
            error=error,
            metadata=StepMetadata(other=dict(reward=reward, truncated=truncated, info=info)),
        )
        observation.metadata.other["screenshot_path"] = screen_path
        observation.metadata.other["env_finished"] = terminated
        return observation

    def html_to_markdown(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        content = markdownify.MarkdownConverter(strip=["img"]).convert_soup(soup)
        content = re.sub(r"\n\s+", "\n", content)
        content = re.sub(r"\n\n+", "\n\n", content)
        return content

    def format_error(self, err: str) -> str:
        logs_separator = "Call log:"
        if logs_separator in err:
            err, logs = err.split(logs_separator)
            logs = "\n".join(logs.split("\n")[:10])
            err = err + f"\n{logs_separator}\n{logs}"
        if not err:
            err = ""
        return err

    def scroll(self, direction: str) -> PageObservation:
        if direction == "down" and self._current_viewport < self._n_viewports:
            self._current_viewport += 1
        elif direction == "up" and self._current_viewport > 1:
            self._current_viewport -= 1
        page = self._current_page[
            self.viewport_size * (self._current_viewport - 1) : self.viewport_size * self._current_viewport
        ]
        return PageObservation(text=page, current_page=self._current_viewport, total_pages=self._n_viewports)

    def goto_page(self, action: GotoPageAction) -> PageObservation:
        # if the URL is a local file or a PDF, playwright cannot open it directly, so we read the content
        if action.url.startswith("file://") or action.url.startswith("/"):
            text, error = read_document(action.url)
            return PageObservation(
                text=self.get_viewport(text),
                current_page=self._current_viewport,
                total_pages=self._n_viewports,
                error=error,
            )
        obs = self.run_browser_action(f"goto('{action.url}')")
        if obs.error:
            text, error = download_file(action.url)
            obs = PageObservation(
                text=self.get_viewport(text),
                current_page=self._current_viewport,
                total_pages=self._n_viewports,
                error=error,
            )
        return obs

    def click(self, action: ClickAction) -> PageObservation:
        self.run_browser_action(f"click('{action.bid}', button='{action.button}', modifiers={action.modifiers})")
        sleep(self.page_load_time_sec)  # wait for the page to load in case click triggers a page change
        return self.run_browser_action("noop()")

    def select_option(self, action: SelectOptionAction) -> PageObservation:
        return self.run_browser_action(f"select_option('{action.bid}', '{action.option}')")

    def hover(self, action: HoverAction) -> PageObservation:
        return self.run_browser_action(f"hover('{action.bid}')")

    def input_text(self, action: InputTextAction) -> PageObservation:
        text = action.text.replace("'", "\\'")
        return self.run_browser_action(f"fill('{action.bid}', '{text}')")

    def press(self, action: PressAction) -> PageObservation:
        return self.run_browser_action(f"press('{action.bid}', '{action.key_comb}')")

    def tab_focus(self, action: TabFocusAction) -> PageObservation:
        return self.run_browser_action(f"tab_focus({action.index})")

    def new_tab(self, action: NewTabAction) -> PageObservation:
        return self.run_browser_action("new_tab()")

    def close_tab(self, action: CloseTabAction) -> PageObservation:
        return self.run_browser_action("tab_close()")

    def go_back(self, action: GoBackAction) -> PageObservation:
        return self.run_browser_action("go_back()")

    def go_forward(self, action: GoForwardAction) -> PageObservation:
        return self.run_browser_action("go_forward()")

    def next_page(self) -> PageObservation:
        return self.scroll("down")

    def get_viewport(self, content: str) -> str:
        self._current_page = content
        self._n_viewports = len(self._current_page) // self.viewport_size + 1
        self._current_viewport = 1
        return self._current_page[
            self.viewport_size * (self._current_viewport - 1) : self.viewport_size * self._current_viewport
        ]


def flatten_axtree(
    AX_tree,
    extra_properties: dict | None = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    filter_visible_only: bool = True,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
    ignored_properties=IGNORED_AXTREE_PROPERTIES,
    ignore_navigation: bool = False,
    hide_bid_if_invisible: bool = False,
    hide_all_children: bool = False,
    nodes_with_bid: list[str] = NODES_WITH_BID,
) -> str:
    """Formats the accessibility tree into a string text"""
    ignored_roles = ["complementary", "navigation"] if ignore_navigation else []
    extra_properties = extra_properties or {}
    node_id_to_idx = {}
    for idx, node in enumerate(AX_tree["nodes"]):
        node_id_to_idx[node["nodeId"]] = idx

    def dfs(node_idx: int, depth: int, parent_node_filtered: bool) -> str:
        tree_str = ""
        node = AX_tree["nodes"][node_idx]
        indent = "  " * depth
        skip_node = False
        filter_node = False
        node_role = node["role"]["value"]

        if node_role in ignored_roles:
            return tree_str
        elif "name" not in node:
            skip_node = True
            pass
        else:
            node_name = node["name"]["value"]
            if "value" in node and "value" in node["value"]:
                node_value = node["value"]["value"]
            else:
                node_value = None

            attributes = []
            bid = node.get("browsergym_id", None)
            for property in node.get("properties", []):
                if "value" not in property:
                    continue
                if "value" not in property["value"]:
                    continue

                prop_name = property["name"]
                prop_value = property["value"]["value"]

                if prop_name == "browsergym_id":
                    bid = prop_value
                elif prop_name in ignored_properties:
                    continue
                elif prop_name in ("required", "focused", "atomic"):
                    if prop_value:
                        attributes.append(prop_name)
                else:
                    attributes.append(f"{prop_name}={repr(prop_value)}")

            if node_role == "generic" and not attributes:
                skip_node = True
            elif node_role != "StaticText":
                filter_node, extra_attributes_to_print = _process_bid(
                    bid,
                    extra_properties=extra_properties,
                    with_visible=with_visible,
                    with_clickable=with_clickable,
                    with_center_coords=with_center_coords,
                    with_bounding_box_coords=with_bounding_box_coords,
                    with_som=with_som,
                    filter_visible_only=filter_visible_only,
                    filter_with_bid_only=filter_with_bid_only,
                    filter_som_only=filter_som_only,
                    coord_decimals=coord_decimals,
                )

                # if either is True, skip the node
                skip_node = skip_node or filter_node or (hide_all_children and parent_node_filtered)

                # insert extra attributes before regular attributes
                attributes = extra_attributes_to_print + attributes

            # actually print the node string
            if not skip_node:
                if node_role == "paragraph":
                    node_str = ""
                elif node_role == "StaticText":
                    node_str = node_name.strip()
                else:
                    node_repr = node_name.strip()
                    if node_repr and node_role != "checkbox":
                        node_str = f"{node_role} {node_repr}"
                    else:
                        node_str = "-" if node_role == "listitem" else node_role
                    if (
                        not (
                            bid is None
                            or (hide_bid_if_invisible and extra_properties.get(bid, {}).get("visibility", 0) < 0.5)
                        )
                        and node_role in nodes_with_bid
                    ):
                        node_str = f"BID:{bid} " + node_str

                if node_value is not None:
                    node_str += f' value={repr(node["value"]["value"])}'

                if attributes:
                    node_str += ", ".join([""] + attributes)

                if "'Advertisement'" in node_str:
                    return tree_str
                tree_str += f"{indent}{node_str}"

        for child_node_id in node["childIds"]:
            if child_node_id not in node_id_to_idx or child_node_id == node["nodeId"]:
                continue
            # mark this to save some tokens
            child_depth = depth if skip_node else (depth + 1)
            child_str = dfs(node_id_to_idx[child_node_id], child_depth, parent_node_filtered=filter_node or skip_node)
            if child_str and node_role != "link":
                if tree_str:
                    tree_str += "\n"
                tree_str += child_str

        return tree_str

    return dfs(0, 0, False)


def download_file(url: str):
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    response = requests.get(url, headers={"User-Agent": user_agent})
    response.raise_for_status()
    return read_document(response)
