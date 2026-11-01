"""B 站视频搜索：为答案附带相关讲解视频，接口失败时优雅降级为站内搜索链接。

B 站搜索接口需要 buvid3 cookie（访问一次首页即可拿到），因此用 Session 懒加载缓存。
"""

from __future__ import annotations

import re

import requests

_UA = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Referer": "https://www.bilibili.com/",
}

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(_UA)
        try:
            s.get("https://www.bilibili.com/", timeout=10)  # 拿 buvid3 cookie
        except requests.RequestException:
            pass
        _session = s
    return _session


def _strip(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _https(u: str) -> str:
    u = u or ""
    if u.startswith("//"):
        return "https:" + u
    return u


def search(keyword: str, top_n: int = 3) -> dict:
    """返回 {"videos": [...], "fallback": bool, "search_url": str}。"""
    search_url = "https://search.bilibili.com/all?keyword=" + keyword
    try:
        resp = _get_session().get(
            "https://api.bilibili.com/x/web-interface/search/type",
            params={"search_type": "video", "keyword": keyword,
                    "page": 1, "page_size": top_n},
            timeout=10,
        )
        payload = resp.json()
        if payload.get("code") != 0:
            return {"videos": [], "fallback": True, "search_url": search_url}

        videos = []
        for v in payload.get("data", {}).get("result") or []:
            bvid = v.get("bvid") or ""
            videos.append({
                "title": _strip(v.get("title")),
                "author": _strip(v.get("author")),
                "play": v.get("play"),
                "cover": _https(v.get("pic")),
                "url": f"https://www.bilibili.com/video/{bvid}" if bvid else (v.get("arcurl") or ""),
            })
        if videos:
            return {"videos": videos, "fallback": False, "search_url": search_url}
        return {"videos": [], "fallback": True, "search_url": search_url}
    except requests.RequestException:
        return {"videos": [], "fallback": True, "search_url": search_url}
