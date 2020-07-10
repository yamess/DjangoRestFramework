from .models import PredictionModel, NewsDataModel
from rest_framework import serializers

class NewsDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = NewsDataModel
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionModel
        fields = '__all__'

