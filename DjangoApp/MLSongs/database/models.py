from django.db import models

class MLModel(models.Model):
    name = models.CharField(max_length=100)
    path = models.CharField(max_length=500)

    def __str__(self):
        return self.name


class Song(models.Model):
    author = models.ForeignKey(MLModel, on_delete=models.CASCADE)
    title = models.CharField(max_length=100)
    audio_file = models.FileField()
    path = models.CharField(max_length=500)

    def __str__(self):
        return self.title


if __name__ == '__main__':
    #MODEL inits here
    m = MLModel(name='MarkovChain')
    m.save()
