---
layout: default
title: Knowledge
---

## 📚 List of Notes | 笔记列表

<ul style="list-style-type: none; padding-left: 0;">
  {% assign notes = site.pages | where: "dir", "/knowledge/" | sort: "date" | reverse %}
  {% for note in notes %}
    {% if note.title and note.title != "Knowledge" %}
      <li style="margin-bottom: 15px;">
        <span style="color: #888; font-size: 0.9em; min-width: 110px; display: inline-block;">
          {{ note.date | date: "%Y-%m-%d" }}
        </span>
        <a href="{{ note.url | relative_url }}" style="font-size: 1.1em;">
          {{ note.title }}
        </a>
      </li>
    {% endif %}
  {% endfor %}
</ul>
