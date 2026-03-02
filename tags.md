---
layout: default
title: Tags
permalink: /tags/
---

{% assign tags = site.tags | sort %}
{% for tag in tags %}
<h3 id="{{ tag[0] | slugify }}">#{{ tag[0] }}</h3>
<ul>
  {% for post in tag[1] %}
    <li>{{ post.date | date: "%Y-%m-%d" }} — <a href="{{ post.url | relative_url }}">{{ post.title }}</a></li>
  {% endfor %}
</ul>
{% endfor %}
