from MLSongs.database.models import MLModel, Song
from rest_framework import serializers

class SongSerializer(serializers.ModelSerializer):

    author = serializers.ReadOnlyField(source='author.name')

    class Meta:
        model = Song
        fields = ['title', 'author', 'audio_file','path']