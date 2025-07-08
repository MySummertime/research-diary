---
layout: default
title: Posts
---

## 📝 List of Posts | 博客列表

<ul style="list-style-type: none; padding-left: 0;">
  {% assign posts_list = site.pages | where: "dir", "/posts/" | sort: "date" | reverse %}
  {% for post in posts_list %}
    {% if post.title and post.title != "Posts" %}
      <li style="margin-bottom: 15px;">
        <span style="color: #888; font-size: 0.9em; min-width: 110px; display: inline-block;">
          {{ post.date | date: "%Y-%m-%d" }}
        </span>
        <a href="{{ post.url | relative_url }}" style="font-size: 1.1em;">
          {{ post.title }}
        </a>
      </li>
    {% endif %}
  {% endfor %}
</ul>
