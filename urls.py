from django.urls import path

from . import views, web

urlpatterns = [
    path("compare/", web.compare_page, name="sig-compare"),     # upload page (browser)
    path("enroll/", views.enroll, name="sig-enroll"),
    path("search/", views.search, name="sig-search"),
    path("verify/", views.verify, name="sig-verify"),
]
