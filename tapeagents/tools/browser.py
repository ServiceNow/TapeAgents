import asyncio
import logging
import os
import re
import threading
import time
from time import sleep
from typing import Any, Callable, Literal
from uuid import uuid4

import gymnasium as gym
import markdownify
import nest_asyncio
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
from bs4 import BeautifulSoup, Comment
from PIL import Image
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
from pydantic import Field

from tapeagents.core import Action, Observation, StepMetadata
from tapeagents.steps import ActionExecutionFailure, ImageObservation
from tapeagents.tools.base import StatefulTool, Tool
from tapeagents.tools.converters import FileConverter
from tapeagents.tools.document_reader import read_document
from tapeagents.tools.grounding import GroundingModel
from tapeagents.tools.simple_browser import PageDownAction, PageObservation, PageUpAction

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
logger = logging.getLogger(__name__)
try:
    nest_asyncio.apply()
except Exception as e:
    logger.warning("Cannot apply nest_asyncio, continuing without it: %s", e)


class OpenUrlAction(Action):
    """
    Action that opens a page with the provided URL and returns its first page content.
    """

    kind: Literal["open_url_action"] = "open_url_action"
    url: str = Field(description="URL to navigate to")


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


class PressAction(Action):
    """
    Action that focuses on an element with a given BID and presses a combination of keys.
    Accepts logical key names: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape,
    ArrowDown, End, Enter, Home, Insert, PageDown, PageUp, ArrowRight, ArrowUp, F1-F12, Digit0-Digit9, KeyA-KeyZ, etc.
    Supported modifier keys: Shift, Control, Alt, Meta.
    """

    kind: Literal["press_action"] = "press_action"
    bid: str = Field(description="BID of the element to focus")
    key_comb: str = Field(description="key combination to press")


class InputTextAction(Action):
    """
    Action that fills in an input element identified by BID with the provided text
    """

    kind: Literal["input_text_action"] = "input_text_action"
    bid: str = Field(description="BID of the input element to fill")
    text: str = Field(description="text to enter into the element")


class TypeTextAction(Action):
    """
    Action that fills in the currently selected input element.
    Only use after clicking on an input element.
    """

    kind: Literal["type_text_action"] = "type_text_action"
    text: str = Field(description="text to enter into the element")


class HoverAction(Action):
    """
    Action that hovers over an element on the page with the provided BID
    """

    kind: Literal["hover_action"] = "hover_action"
    bid: str = Field(description="BID of the element to hover over")


class MouseHoverAction(Action):
    """
    Action that hovers over an icon or control on the screen
    """

    kind: Literal["mouse_hover_action"] = "mouse_hover_action"
    element_description: str = Field(description="brief description of the element to hover over")


class SelectOptionAction(Action):
    """
    Action that selects an option in a dropdown or combobox element with the provided BID.
    ONLY applicable to dropdown and combobox elements!
    """

    kind: Literal["select_option_action"] = "select_option_action"
    bid: str = Field(description="BID of the dropdown or combobox to select from")
    element_description: str = Field(description="brief description of the dropdown or combobox")
    option: str = Field(description="option to select")


class ClickBIDAction(Action):
    """
    Action that clicks the element on the page with the provided BID
    """

    kind: Literal["click_bid_action"] = "click_bid_action"
    bid: str = Field(description="BID of the element to click")
    button: Literal["left", "middle", "right"] = Field(description="button to click", default="left")
    modifiers: list[Literal["Alt", "Control", "Meta", "Shift"]] = Field(
        description="modifier keys to press", default_factory=list
    )


class ClickElementAction(Action):
    """
    Action that clicks an element on the screen.
    When mentioning a date in the element description, use the format commonly spoken or written by humans,
    such as "2 February 2025," rather than machine-readable formats. The day should come before the month,
    and the year should be written in full (e.g., "3 November 2023" instead of "2023-11-03").
    Only describe one specific element that is currently visible on the screen!
    """

    kind: Literal["click_element_action"] = "click_element_action"
    element_description: str = Field(description="brief description of the element to click")


class ClickCoordinatesAction(Action):
    """
    Action that moves the mouse to a (x, y) coordinate location and click a mouse button.
    """

    kind: Literal["click_coordinates_action"] = "click_coordinates_action"
    x: float = Field(description="x coordinate of the click")
    y: float = Field(description="y coordinate of the click")
    button: Literal["left", "middle", "right"] = Field(description="button to click", default="left")


class PageScreenshotObservation(ImageObservation):
    kind: Literal["page_screenshot_observation"] = "page_screenshot_observation"


class Browser(StatefulTool):
    """
    Browser tool that can load web pages and interact with their content.
    """

    actions: tuple[type[Action], ...] = ()  # will get assigned in model_post_init
    observations: tuple[type[Observation], ...] = (PageObservation, PageScreenshotObservation)
    tab_actions: list[type[Action]] = [CloseTabAction, NewTabAction, TabFocusAction]
    observation_format: Literal["axtree", "html", "markdown_html"] = "axtree"
    use_grounding: bool = False
    navigation_only: bool = False
    viewport_chars: int = 32000
    viewport_height: int = 720
    viewport_width: int = 1024
    timeout_ms: int = 30000
    headless: bool = True
    save_video: bool = False
    exp_path: str | None = None
    page_load_time_sec: int = 1
    gym_kwargs: dict = {}
    gym_task: str = "browsergym/openended"
    mock: bool = False # TODO: investigate why renaming it to any other name brokes running 2 async tests at one pytest call (but not a singular one)

    _env: BrowserEnv = None  # type: ignore
    _current_page: str = ""
    _current_viewport: int = 0
    _n_viewports: int = 1
    _action_map: dict[type[Action], Callable] = {}
    _traces_dir: str | None = None
    _record_video_dir: str | None = None
    _screenshots_dir: str | None = None
    _task_id: str = ""
    _grounding: GroundingModel | None = None
    _last_image: Image.Image | None = None
    _non_browser_doc: bool = False

    def model_post_init(self, __context: Any):
        self._current_page = ""
        self._current_viewport = 1
        self._n_viewports = 1
        if self.use_grounding:
            self._grounding = GroundingModel()
            logger.info("Using grounding model")
            self._action_map = {
                ClickElementAction: self.click_grounded,
                CloseTabAction: self.close_tab,
                TypeTextAction: self.input_text_grounded,
                GoBackAction: self.go_back,
                GoForwardAction: self.go_forward,
                OpenUrlAction: self.goto_page,
                MouseHoverAction: self.hover_grounded,
                NewTabAction: self.new_tab,
                PageDownAction: self.next_page,
                PageUpAction: self.previous_page,
                TabFocusAction: self.tab_focus,
            }
        elif self.navigation_only:
            logger.info("Navigation only mode")
            self._action_map = {
                ClickBIDAction: self.click_bid,
                GoBackAction: self.go_back,
                GoForwardAction: self.go_forward,
                OpenUrlAction: self.goto_page,
                PageDownAction: self.next_page,
                PageUpAction: self.previous_page,
            }
        else:
            self._action_map = {
                ClickBIDAction: self.click_bid,
                ClickCoordinatesAction: self.click_coordinates,
                SelectOptionAction: self.select_option,
                InputTextAction: self.input_text,
                PressAction: self.press,
                GoBackAction: self.go_back,
                GoForwardAction: self.go_forward,
                OpenUrlAction: self.goto_page,
                HoverAction: self.hover,
                PageDownAction: self.next_page,
                PageUpAction: self.previous_page,
            }
        self.actions = tuple(self._action_map.keys())
        if self.exp_path:
            assert os.path.isdir(self.exp_path)
            self._traces_dir = os.path.join(self.exp_path, "playwright_traces")
            self._record_video_dir = os.path.join(self.exp_path, "attachments/browser/videos")
            self._screenshots_dir = os.path.join(self.exp_path, "attachments/browser/screenshots")
            os.makedirs(self._traces_dir, exist_ok=True)
            os.makedirs(self._record_video_dir, exist_ok=True)
            os.makedirs(self._screenshots_dir, exist_ok=True)
        if self.mock:
            logger.info("Browser initialized")
            return
        self._env = gym.make(
            self.gym_task,
            headless=self.headless,
            record_video_dir=self._record_video_dir if self.save_video else None,
            action_mapping=HighLevelActionSet(demo_mode="default", subsets=["coord", "workarena++"]).to_python_code,
            timeout=self.timeout_ms,
            viewport={"width": self.viewport_width, "height": self.viewport_height},
            task_kwargs={"start_url": "about:blank"},
            **self.gym_kwargs,
        )  # type: ignore
        while not isinstance(self._env, BrowserEnv):
            self._env = self._env.env
        self._env.reset()
        self._env.context.tracing.start(screenshots=True, snapshots=True)
        screenshot = self._env.step("noop()")[0]["screenshot"]
        self._save_last_screenshot(screenshot)
        logger.info("Browser and gym initialized")

    def execute_action(self, action: Action) -> PageObservation | PageScreenshotObservation:
        action_type = type(action)
        if action_type in self._action_map:
            return self._action_map[action_type](action)
        raise ValueError(f"Unknown action: {action_type}")

    def start_task(self, task_id: str, seed: int = 1, **kwargs) -> dict:
        self._task_id = task_id
        t = time.perf_counter()
        if self._env is not None:
            self._env.close()
        logger.info(f"Old gym close took {time.perf_counter() - t:.2f}s")
        t = time.perf_counter()
        self._env = gym.make(
            task_id,
            headless=self.headless,
            record_video_dir=self._record_video_dir if self.save_video else None,
            action_mapping=HighLevelActionSet(demo_mode="default", subsets=["coord", "workarena++"]).to_python_code,
            timeout=self.timeout_ms,
            **kwargs,
        )  # type: ignore
        logger.info(f"New gym make took {time.perf_counter() - t:.2f}s")
        t = time.perf_counter()
        start_obs, info = self._env.reset(seed=seed)
        logger.info(f"Gym reset took {time.perf_counter() - t:.2f}s")
        self._env.unwrapped.context.tracing.start(screenshots=True, snapshots=True)
        self._env.unwrapped.chat.add_message(role="assistant", msg="Running TapeAgent...")
        assert self._env.unwrapped.task is not None
        info = {
            "name": self._env.unwrapped.task.get_task_id(),
            "goal": start_obs["goal"],
            "task_info": info["task_info"],
            "video": "",
            "chat_video": "",
        }
        return info

    def close(self):
        assert self._traces_dir is not None
        self._env.unwrapped.context.tracing.stop(path=os.path.join(self._traces_dir, f"{self._task_id}.zip"))
        self._env.close()

    def _save_last_screenshot(self, image) -> str:
        if self._screenshots_dir is None:
            return ""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        if image.size[0] != self.viewport_width or image.size[1] != self.viewport_height:
            image = image.resize((self.viewport_width, self.viewport_height))
        self._last_image = image
        pic_uid = uuid4().hex
        img_path = os.path.join(self._screenshots_dir, f"{pic_uid}.png")
        image.save(img_path)
        return img_path

    def reset(self):
        self._env.step("goto('about:blank')")

    def run_browser_action(self, action_text: str) -> PageObservation:
        obs_dict, reward, terminated, truncated, info = self._env.step(action_text)
        error = self.format_error(obs_dict["last_action_error"])
        screen_path = self._save_last_screenshot(obs_dict["screenshot"])
        if self.use_grounding:
            observation = PageScreenshotObservation(image_path=screen_path)
        else:
            if error:
                content = ""
            elif self.observation_format == "axtree":
                content = f"=== ACCESSIBILITY TREE ===\n{flatten_axtree(obs_dict['axtree_object'])}"
            elif self.observation_format == "html":
                content = f"=== HTML ===\n{prune_html(flatten_dom_to_str(obs_dict['dom_object']))}"
            elif self.observation_format == "markdown_html":
                html_content = prune_html(flatten_dom_to_str(obs_dict["dom_object"]))
                content = f"=== MARKDOWN ===\n{self.html_to_markdown(html_content)}"
            else:
                raise ValueError(f"Unknown observation format: {self.observation_format}")

            observation = PageObservation(
                text=self.get_viewport(content),
                current_page=self._current_viewport,
                total_pages=self._n_viewports,
                error=error,
                metadata=StepMetadata(other=dict(reward=reward, truncated=truncated, info=info)),
            )
        if self._screenshots_dir is not None:
            observation.metadata.other["screenshot_path"] = os.path.relpath(screen_path, self._screenshots_dir)
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

    def goto_page(self, action: OpenUrlAction) -> PageObservation:
        # if the URL is a local file or a PDF, playwright cannot open it directly, so we read the content
        if action.url.startswith("file://") or action.url.startswith("/"):
            text, error = read_document(action.url)
            self._non_browser_doc = True
            return PageObservation(
                text=self.get_viewport(text),
                current_page=self._current_viewport,
                total_pages=self._n_viewports,
                error=error,
            )
        self._non_browser_doc = False
        obs = self.run_browser_action(f"goto('{action.url}')")
        return obs

    def click_bid(self, action: ClickBIDAction) -> PageObservation:
        try:
            modifiers = [m for m in action.modifiers if m in ["Alt", "Control", "Meta", "Shift"]]
            self.run_browser_action(f"click('{action.bid}', button='{action.button}', modifiers={modifiers})")
        except Exception as e:
            logger.warning(f"Click failed: {e}")
        sleep(self.page_load_time_sec)  # wait for the page to load in case click triggers a page change
        return self.run_browser_action("noop()")

    def click_coordinates(self, action: ClickCoordinatesAction) -> PageObservation:
        self.run_browser_action(f"mouse_click({action.x}, {action.y}, button='{action.button}')")
        sleep(self.page_load_time_sec)  # wait for the page to load in case click triggers a page change
        return self.run_browser_action("noop()")

    def click_grounded(self, action: ClickElementAction) -> PageObservation:
        x, y = self._grounding.get_coords(self._last_image, f"click at {action.element_description}")
        logger.info(f"Click at {action.element_description}: {x}, {y}")
        return self.run_browser_action(f"mouse_click({x}, {y})")

    def select_option(self, action: SelectOptionAction) -> PageObservation:
        return self.run_browser_action(f"select_option('{action.bid}', '{action.option}')")

    def hover(self, action: HoverAction) -> PageObservation:
        return self.run_browser_action(f"hover('{action.bid}')")

    def hover_grounded(self, action: MouseHoverAction) -> PageObservation:
        x, y = self._grounding.get_coords(self._last_image, f"click at {action.element_description}")
        logger.info(f"Move cursor at {action.element_description}: {x}, {y}")
        return self.run_browser_action(f"mouse_move({x}, {y})")

    def input_text(self, action: InputTextAction) -> PageObservation:
        text = action.text.replace("'", "\\'")
        return self.run_browser_action(f"fill('{action.bid}', '{text}')")

    def input_text_grounded(self, action: TypeTextAction) -> PageObservation:
        text = action.text.replace("'", "\\'")
        return self.run_browser_action(f"keyboard_type('{text}')")

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

    def next_page(self, action) -> PageObservation:
        if self._current_viewport < self._n_viewports:
            self._current_viewport += 1
            if not self._non_browser_doc:
                self._env.step(f"scroll(0, {self.viewport_height})")
            page = self._current_page[
                self.viewport_chars * (self._current_viewport - 1) : self.viewport_chars * self._current_viewport
            ]
            obs = PageObservation(text=page, current_page=self._current_viewport, total_pages=self._n_viewports)
        else:
            obs = PageObservation(
                text="No more pages to scroll", current_page=self._current_viewport, total_pages=self._n_viewports
            )
        return obs

    def previous_page(self, action) -> PageObservation:
        if self._current_viewport > 1:
            self._current_viewport -= 1
            if not self._non_browser_doc:
                self._env.step(f"scroll(0, -{self.viewport_height})")
            page = self._current_page[
                self.viewport_chars * (self._current_viewport - 1) : self.viewport_chars * self._current_viewport
            ]
            obs = PageObservation(text=page, current_page=self._current_viewport, total_pages=self._n_viewports)
        else:
            obs = PageObservation(
                text="Already at the top page", current_page=self._current_viewport, total_pages=self._n_viewports
            )
        return obs

    def get_viewport(self, content: str) -> str:
        self._current_page = content
        self._n_viewports = len(self._current_page) // self.viewport_chars + 1
        self._current_viewport = 1
        return self._current_page[
            self.viewport_chars * (self._current_viewport - 1) : self.viewport_chars * self._current_viewport
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
                    node_str += f" value={repr(node['value']['value'])}"

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


headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}


def minimize_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove script and style tags
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Strip attributes from all tags except href and src
    for tag in soup.find_all(True):  # Find all tags
        attrs = tag.attrs
        # Save href and src if they exist
        href = attrs.get("href", None)
        src = attrs.get("src", None)
        alt = attrs.get("alt", None)
        # Clear all attributes
        tag.attrs = {}
        # Restore href and src if they existed
        if href:
            tag["href"] = href
        if src:
            tag["src"] = src
        if alt:
            tag["alt"] = alt
        # Remove tag if it has no content and its not an image
        if not tag.string and not tag.contents and tag.name != "img":
            tag.decompose()

    for tag in soup.find_all(True):
        if not tag.string and not tag.contents and tag.name != "img":
            tag.decompose()

    minimal_html = str(soup)
    # Remove extra whitespace
    minimal_html = re.sub(r"\s+", " ", minimal_html)
    # Remove empty lines
    minimal_html = re.sub(r"\n\s*\n", "\n", minimal_html)
    return minimal_html


class Tls(threading.local):
    sync: bool = True

    def __init__(self) -> None:
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
        except Exception:
            self.sync = False
            logger.warning("Using playwright async")
            self.playwright = asyncio.run(async_playwright().start())
            self.browser = asyncio.run(self.playwright.chromium.launch(headless=True))


class Fetcher(Tool):
    """
    A minimal text-based web browser designed for AI agent use.
    Can read web pages, PDFs and other document types.
    """

    action: type[Action] = OpenUrlAction
    observation: type[Observation] = PageObservation
    cached: bool = True
    width: int = Field(default=1280, description="Width of the browser window")
    height: int = Field(default=800, description="Height of the browser window")
    sleep_time: int = Field(default=2, description="Time to wait for the page to load")
    timeout: int = Field(default=20, description="Timeout for the request in seconds")

    _tls: Tls | None = None  # Use a private instance variable

    def model_post_init(self, __context: Any):
        # Initialize tls instance here
        self._tls = Tls()

    @property
    def tls(self) -> Tls:
        if self._tls is None:
            # defensive, should be initialized in model_post_init
            self._tls = Tls()
        return self._tls

    def fetch_for_llm(self, url: str) -> tuple[str, str]:
        try:
            action = OpenUrlAction(url=url)
            obs = self.run(action)
            if isinstance(obs, ActionExecutionFailure):
                raise Exception(ActionExecutionFailure.error)
            text = obs.text if isinstance(obs, PageObservation) else ""
        except Exception as e:
            logger.exception(f"Failed to fetch page {url}: {e}")
            text = ""
        return url, text

    def execute_action(self, action: OpenUrlAction) -> PageObservation:
        """
        Fetches the content of a URL and returns it as a string in a format suitable for LLMs.
        If the content type is HTML, it minimizes the HTML content.
        If the content type is plain text, it returns the text as is.
        In all other cases, it uses the FileConverter to convert the response to text.
        """
        url = action.url
        if url.endswith(".pdf"):
            content_type = "application/pdf"
        else:
            try:
                content_type = requests.head(url, headers=headers).headers.get("content-type", "")
            except Exception as e:
                logger.exception(f"Error fetching headers for {url}, interpret as html page: {e}")
                content_type = "text/html"
        if "text/html" in content_type.lower():
            if self.tls.sync:
                html_content = self.get_html(url)
            else:
                html_content = asyncio.run(self.async_get_html(url))
            text = minimize_html(html_content)
        elif "text/plain" in content_type.lower():
            response = requests.get(url, headers=headers)
            text = response.text
        else:
            response = requests.get(url, headers=headers)
            result = FileConverter().convert_response(response)
            text = result.text_content
        return PageObservation(text=text, current_page=1, total_pages=1)

    def get_html(self, url: str) -> str:
        page = self.tls.browser.new_page(viewport={"width": self.width, "height": self.height})
        page.goto(url, timeout=self.timeout * 1000)
        sleep(self.sleep_time)  # Wait for page to load and render
        html_content = page.content()
        page.close()
        return html_content

    async def async_get_html(self, url: str) -> str:
        page = await self.tls.browser.new_page(viewport={"width": self.width, "height": self.height})
        await page.goto(url, timeout=self.timeout * 1000)
        await asyncio.sleep(self.sleep_time)  # Wait for page to load and render
        html_content = await page.content()
        await page.close()
        return html_content
