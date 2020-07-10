from django.http import Http404
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import PredictionModel, NewsDataModel
from .serializers import PredictionSerializer, NewsDataSerializer
from .fakenews.engine import preprocess, onnx_predict
from .apps import FakenewsappConfig

# Create your views here.
class NewsDataViewList(APIView):
    def get(self, request):
        newsdata = NewsDataModel.objects.all()
        serializer = NewsDataSerializer(newsdata, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        serializer = NewsDataSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class NewsDataDetail(APIView):
    def get_object(self, pk):
        try:
            return NewsDataModel.objects.get(pk=pk)
        except NewsDataModel.DoesNotExist:
            raise Http404

    def get(self, request, pk):
        newsdata = self.get_object(pk)
        serializer = NewsDataSerializer(newsdata)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def delete(self, request, pk):
        newsdata = self.get_object(pk)
        newsdata.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

# @api_view(['POST'])
# def predictor(request):

class PredictView(APIView):
    def post(self, request):
        newsDataSerializer = NewsDataSerializer(data=request.data)
        if newsDataSerializer.is_valid():
            title = request.data.get("title")
            text = request.data.get("text")

            model_input = preprocess(
                title=title,
                text=text,
                tokenizer=FakenewsappConfig.tokenizer
            )

            prediction = onnx_predict(
                inputs=model_input,
                session=FakenewsappConfig.session
            )

            # newsDataSerializer.save()
            newsData = NewsDataModel(
                title=newsDataSerializer.validated_data["title"],
                text=newsDataSerializer.validated_data["text"]
            )
            # newsData.save()

            predictedModel = PredictionModel(
                logits=prediction["logits"],
                predictionValue=prediction["predictedValue"],
                predictionText=prediction["predictedText"],
                predictionProb=prediction["predictedProb"],
                dataId=newsData
            )
            predictedSerializer = PredictionSerializer(predictedModel)
            newsData.save()
            predictedModel.save()

            return Response(predictedSerializer.data, status=status.HTTP_200_OK)
        return Response(status=status.HTTP_400_BAD_REQUEST)

class PredictionList(APIView):
    def get(self, request):
        prediction = PredictionModel.objects.all()
        predictionSerializer = PredictionSerializer(prediction, many=True)
        return Response(predictionSerializer.data, status=status.HTTP_200_OK)
