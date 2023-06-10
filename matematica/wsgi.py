"""
WSGI config for matematica project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "matematica.settings")
# os.environ.setdefault("DJANGO_URLS_MODULE", "matematica.matematica.urls")
# os.environ.setdefault("DJANGO_ASGI_MODULE", "matematica.matematica.asgi")

application = get_wsgi_application()

# per vercel: .vercel.app
# app = application