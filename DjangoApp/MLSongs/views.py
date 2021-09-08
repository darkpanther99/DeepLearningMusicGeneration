from django.shortcuts import render
from MLSongs.models import MLModel, Song
from rest_framework import viewsets
from MLSongs.serializers import SongSerializer

class SongViewSet(viewsets.ModelViewSet):
    queryset = Song.objects.all()
    serializer_class = SongSerializer