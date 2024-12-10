import os
from time import sleep
from typing import Literal
from uuid import uuid4

import gymnasium as gym
import numpy as np
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.env import BrowserEnv
from browsergym.utils.obs import IGNORED_AXTREE_PROPERTIES, _process_bid
from PIL import Image
from pydantic import Field

from tapeagents.core import Action, Observation, StepMetadata

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


class PageObservation(Observation):
    kind: Literal["page_observation"] = "page_observation"
    text: str
    current_page: int
    total_pages: int
    last_action_error: str = ""


class GotoPageAction(Action):
    """
    Action that opens the page with the provided URL in the current tab
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
    Action that scrolls the page in the provided direction
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


class Browser:
    _env: BrowserEnv
    actions: list[type[Action]] = [
        ClickAction,
        GotoPageAction,
        GoBackAction,
        GoForwardAction,
        HoverAction,
        InputTextAction,
        PressAction,
        ScrollAction,
        SelectOptionAction,
    ]
    tab_actions: list[type[Action]] = [CloseTabAction, NewTabAction, TabFocusAction]

    def __init__(
        self,
        viewport_size: int = 64000,
        headless: bool = True,
        log_path: str | None = None,
        page_load_time_sec: int = 2,
    ) -> None:
        self.viewport_size = viewport_size
        self.headless = headless
        self.page_load_time_sec = page_load_time_sec
        self.current_page = ""
        self.current_viewport = 1
        self.n_viewports = 1
        self.log_path = log_path
        self.action_map = {
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
        if log_path:
            assert os.path.isdir(log_path)
            self.traces_dir = os.path.join(log_path, "playwright_traces")
            self.record_video_dir = os.path.join(log_path, "videos")
            self.screenshots_dir = os.path.join(log_path, "screenshots")
            os.makedirs(self.traces_dir, exist_ok=True)
            os.makedirs(self.record_video_dir, exist_ok=True)
            os.makedirs(self.screenshots_dir, exist_ok=True)

    def start_task(self, task_id: str, seed: int, **kwargs) -> dict:
        self._env = gym.make(
            task_id,
            headless=self.headless,
            record_video_dir=self.record_video_dir,
            action_mapping=HighLevelActionSet(demo_mode="default").to_python_code,
            timeout=60000,
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

    def close(self, task_name: str):
        self._env.context.tracing.stop(path=os.path.join(self.traces_dir, f"{task_name}.zip"))
        self._env.close()

    def _screenshot_to_img_file(self, image) -> str:
        if self.screenshots_dir is None:
            return ""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")
        pic_uid = uuid4().hex
        img_path = os.path.join(self.screenshots_dir, f"{pic_uid}.png")
        image.save(img_path)
        return os.path.relpath(img_path, self.screenshots_dir)

    def perform_action(self, action_text: str) -> PageObservation:
        self._env.page.set_default_timeout(60000)
        obs, reward, terminated, truncated, info = self._env.step(action_text)
        last_action_error = self.format_error(obs["last_action_error"])
        accessibility_tree = obs["axtree_object"]
        screen_path = self._screenshot_to_img_file(obs["screenshot"])
        text = self.get_viewport(accessibility_tree)
        obs = PageObservation(
            text=text,
            current_page=self.current_viewport,
            total_pages=self.n_viewports,
            last_action_error=last_action_error,
            metadata=StepMetadata(other=dict(reward=reward, truncated=truncated, info=info)),
        )
        obs.metadata.other["screenshot_path"] = screen_path
        obs.metadata.other["env_finished"] = terminated
        return obs

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
        if direction == "down" and self.current_viewport < self.n_viewports:
            self.current_viewport += 1
        elif direction == "up" and self.current_viewport > 1:
            self.current_viewport -= 1
        page = self.current_page[
            self.viewport_size * (self.current_viewport - 1) : self.viewport_size * self.current_viewport
        ]
        return PageObservation(
            text=page,
            current_page=self.current_viewport,
            total_pages=self.n_viewports,
        )

    def goto_page(self, action: GotoPageAction) -> PageObservation:
        return self.perform_action(f"goto('{action.url}')")

    def click(self, action: ClickAction) -> PageObservation:
        self.perform_action(f"click('{action.bid}', button='{action.button}', modifiers={action.modifiers})")
        sleep(self.page_load_time_sec)  # wait for the page to load in case click triggers a page change
        return self.perform_action("noop()")

    def select_option(self, action: SelectOptionAction) -> PageObservation:
        return self.perform_action(f"select_option('{action.bid}', '{action.option}')")

    def hover(self, action: HoverAction) -> PageObservation:
        return self.perform_action(f"hover('{action.bid}')")

    def input_text(self, action: InputTextAction) -> PageObservation:
        text = action.text.replace("'", "\\'")
        return self.perform_action(f"fill('{action.bid}', '{text}')")

    def press(self, action: PressAction) -> PageObservation:
        return self.perform_action(f"press('{action.bid}', '{action.key_comb}')")

    def tab_focus(self, action: TabFocusAction) -> PageObservation:
        return self.perform_action(f"tab_focus({action.index})")

    def new_tab(self, action: NewTabAction) -> PageObservation:
        return self.perform_action("new_tab()")

    def close_tab(self, action: CloseTabAction) -> PageObservation:
        return self.perform_action("tab_close()")

    def go_back(self, action: GoBackAction) -> PageObservation:
        return self.perform_action("go_back()")

    def go_forward(self, action: GoForwardAction) -> PageObservation:
        return self.perform_action("go_forward()")

    def next_page(self) -> PageObservation:
        return self.scroll("down")

    def get_viewport(self, accessibility_tree: dict) -> str:
        self.current_page = flatten_axtree(accessibility_tree)
        self.n_viewports = len(self.current_page) // self.viewport_size + 1
        self.current_viewport = 1
        return self.current_page[
            self.viewport_size * (self.current_viewport - 1) : self.viewport_size * self.current_viewport
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
