from django.db import models
from django.contrib import admin

# Create your models here.
class NewsDataModel(models.Model):
    title = models.TextField()
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.title)

class PredictionModel(models.Model):
    logits = models.DecimalField(max_digits=10, decimal_places=5)
    predictionValue = models.PositiveSmallIntegerField()
    predictionText = models.CharField(max_length=15)
    predictionProb = models.DecimalField(max_digits=5, decimal_places=4)
    dataId = models.ForeignKey(to=NewsDataModel, on_delete=models.CASCADE)

    def __str__(self):
        return f"{self.predictionText} - {self.predictionValue} - {self.predictionProb}"
