# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:04:08 2021

@author: rukap
"""

import urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
import ssl

import json
sumtillnow=0

data = '''

  [
    {
      "name":"Cohan",
      "count":100
    },
    {
      "name":"Nuriyah",
      "count":99
    },
    {
      "name":"Oluwatamilore",
      "count":97
    },
    {
      "name":"Harold",
      "count":94
    },
    {
      "name":"Gregor",
      "count":90
    },
    {
      "name":"Shea",
      "count":90
    },
    {
      "name":"Teos",
      "count":89
    },
    {
      "name":"Kenan",
      "count":87
    },
    {
      "name":"Jedd",
      "count":86
    },
    {
      "name":"Rio",
      "count":83
    },
    {
      "name":"Kile",
      "count":83
    },
    {
      "name":"Jonson",
      "count":80
    },
    {
      "name":"Veera",
      "count":78
    },
    {
      "name":"Cieran",
      "count":78
    },
    {
      "name":"Kerr",
      "count":77
    },
    {
      "name":"Princess",
      "count":75
    },
    {
      "name":"Johanna",
      "count":74
    },
    {
      "name":"Isa",
      "count":73
    },
    {
      "name":"Elisha",
      "count":68
    },
    {
      "name":"Irmak",
      "count":63
    },
    {
      "name":"Jaina",
      "count":62
    },
    {
      "name":"Elivia",
      "count":60
    },
    {
      "name":"Kelam",
      "count":56
    },
    {
      "name":"Abaigeal",
      "count":56
    },
    {
      "name":"Kiyaleigh",
      "count":52
    },
    {
      "name":"Saul",
      "count":48
    },
    {
      "name":"Guy",
      "count":47
    },
    {
      "name":"Carmyle",
      "count":46
    },
    {
      "name":"Logyn",
      "count":43
    },
    {
      "name":"Azlan",
      "count":42
    },
    {
      "name":"Saghun",
      "count":41
    },
    {
      "name":"Findlie",
      "count":41
    },
    {
      "name":"Khizer",
      "count":33
    },
    {
      "name":"Harley",
      "count":31
    },
    {
      "name":"Amaylyuh",
      "count":30
    },
    {
      "name":"Sethu",
      "count":27
    },
    {
      "name":"Orson",
      "count":27
    },
    {
      "name":"Brendon",
      "count":26
    },
    {
      "name":"Architha",
      "count":23
    },
    {
      "name":"Maha",
      "count":23
    },
    {
      "name":"Ivana",
      "count":21
    },
    {
      "name":"Edyn",
      "count":19
    },
    {
      "name":"Dissanayake",
      "count":19
    },
    {
      "name":"Shinade",
      "count":16
    },
    {
      "name":"Brajan",
      "count":16
    },
    {
      "name":"Thrinei",
      "count":15
    },
    {
      "name":"Atlanta",
      "count":13
    },
    {
      "name":"Teegan",
      "count":13
    },
    {
      "name":"Willa",
      "count":5
    },
    {
      "name":"Rosalind",
      "count":1
    }
  ]
'''

info = json.loads(data)
print('User count:', len(info))

for item in info:
    print('Name', item['name'])
    print('Count', item['count'])
    sumtillnow=sumtillnow+int(item['count'])
print('Sum=', sumtillnow)    