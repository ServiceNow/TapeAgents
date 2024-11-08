import os
from uuid import uuid4

import gymnasium as gym
import numpy as np
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.env import BrowserEnv
from browsergym.utils.obs import IGNORED_AXTREE_PROPERTIES, _process_bid, flatten_axtree_to_str
from browsergym.workarena.tasks.base import AbstractServiceNowTask
from PIL import Image

IGNORED_ROLES = []  # ["contentinfo", "LineBreak", "banner"]  #  "Iframe",
NAVIGATION_ROLES = ["complementary", "navigation"]

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


class GymBrowser:
    env: BrowserEnv

    def __init__(
        self,
        viewport_size: int = 64000,
        headless: bool = True,
        log_path: str | None = None,
    ) -> None:
        self.viewport_size = viewport_size
        self.headless = headless
        self.action_mapping = HighLevelActionSet(demo_mode="default").to_python_code
        self.current_page = ""
        self.current_viewport = 1
        self.n_viewports = 1
        self.log_path = log_path
        if log_path:
            assert os.path.isdir(log_path)
            self.traces_dir = os.path.join(log_path, "playwright_traces")
            self.record_video_dir = os.path.join(log_path, "videos")
            self.screenshots_dir = os.path.join(log_path, "screenshots")
            os.makedirs(self.traces_dir, exist_ok=True)
            os.makedirs(self.record_video_dir, exist_ok=True)
            os.makedirs(self.screenshots_dir, exist_ok=True)

    def start_task(self, task_entrypoint: type[AbstractServiceNowTask], seed: int) -> dict:
        self.env = gym.make(
            f"browsergym/{task_entrypoint.get_task_id()}",
            wait_for_user_message=False,
            headless=self.headless,
            record_video_dir=self.record_video_dir,
            action_mapping=self.action_mapping,
            timeout=60000,
        )  # type: ignore
        start_obs, info = self.env.reset(seed=seed)
        self.env.context.tracing.start(screenshots=True, snapshots=True)
        self.env.chat.add_message(role="assistant", msg="Running TapeAgent...")
        assert self.env.task is not None
        info = {
            "name": self.env.task.get_task_id(),
            "goal": start_obs["goal"],
            "task_info": info["task_info"],
            "video": os.path.basename(self.env.page.video.path()) if self.env.page.video else "",
            "chat_video": os.path.basename(self.env.chat.page.video.path()) if self.env.chat.page.video else "",
        }
        return info

    def close(self, task_name: str):
        self.env.context.tracing.stop(path=os.path.join(self.traces_dir, f"{task_name}.zip"))
        self.env.close()

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

    def perform_action(self, action_text: str, baseline_obs: bool = False) -> tuple[str, str, str, bool]:
        self.env.page.set_default_timeout(60000)
        obs, reward, terminated, truncated, info = self.env.step(action_text)
        last_action_error = self.format_error(obs["last_action_error"])
        accessibility_tree = obs["axtree_object"]
        screen_path = self._screenshot_to_img_file(obs["screenshot"])
        if baseline_obs:
            bid_info = """\
Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions.

"""
            visible_tag_note = """\
Note: You can only interact with visible elements. If the "visible" tag is not
present, the element is not visible on the page.

"""
            axtree_txt = flatten_axtree_to_str(
                obs["axtree_object"],
                extra_properties=obs["extra_element_properties"],
                with_visible=True,
                with_clickable=True,
                with_center_coords=False,
                with_bounding_box_coords=False,
                filter_visible_only=False,
                filter_with_bid_only=False,
                filter_som_only=False,
            )
            ax_tree = f"\n## AXTree:\n{bid_info}{visible_tag_note}{axtree_txt}\n"
            bid = obs["focused_element_bid"]
            bid_str = f"bid={repr(bid)}" if bid else "None"
            focused_element = f"## Focused element:{bid_str}\n"
            error = f"\n## Error from previous action:\n{last_action_error}" if last_action_error else ""
            text = f"""
# Observation of current step:
{ax_tree}{focused_element}{error}
"""
        else:
            text = self.get_viewport(accessibility_tree)
        return text, screen_path, last_action_error, terminated

    def format_error(self, err: str) -> str:
        logs_separator = "Call log:"
        if logs_separator in err:
            err, logs = err.split(logs_separator)
            logs = "\n".join(logs.split("\n")[:10])
            err = err + f"\n{logs_separator}\n{logs}"
        if not err:
            err = ""
        return err

    def scroll(self, direction: str) -> str:
        if direction == "down" and self.current_viewport < self.n_viewports:
            self.current_viewport += 1
        elif direction == "up" and self.current_viewport > 1:
            self.current_viewport -= 1
        return self.current_page[
            self.viewport_size * (self.current_viewport - 1) : self.viewport_size * self.current_viewport
        ]

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
    ignored_roles=IGNORED_ROLES,
    ignored_properties=IGNORED_AXTREE_PROPERTIES,
    ignore_navigation: bool = False,
    hide_bid_if_invisible: bool = False,
    hide_all_children: bool = False,
    nodes_with_bid: list[str] = NODES_WITH_BID,
) -> str:
    """Formats the accessibility tree into a string text"""
    if ignore_navigation:
        ignored_roles += NAVIGATION_ROLES
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
