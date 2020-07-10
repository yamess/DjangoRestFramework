from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('newsdatalist/', views.NewsDataViewList.as_view()),
    path('newsdatadetail/<int:pk>', views.NewsDataDetail.as_view()),
    path('predict/', views.PredictView.as_view()),
    path('predictionlist/', views.PredictionList.as_view())
]
