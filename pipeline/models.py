from django.db import models


class DatasetTrain(models.Model):
    upload = models.FileField(upload_to='uploads/')
    key = models.CharField(max_length=200, unique=True)

    def __str__(self):
        return self.key


class DatasetTest(models.Model):
    upload = models.FileField(upload_to='uploads/')
    key = models.CharField(max_length=200, unique=True)

    def __str__(self):
        return self.key


class DatasetDescription(models.Model):
    categorical_columns = models.CharField(max_length=10000)
    numerical_columns = models.CharField(max_length=10000)
    id_column = models.CharField(max_length=1000)
    label_column = models.CharField(max_length=1000)
    key = models.CharField(max_length=200, unique=True)

    def __str__(self):
        return self.key


class Weight(models.Model):
    key = models.CharField(max_length=200, unique=True)
    weight = models.FileField(upload_to='weights/')

    def __str__(self):
        return self.key


class ImportantFeature(models.Model):
    key = models.CharField(max_length=200, unique=True)
    imp_feat = models.FileField(upload_to='important_features/')

    def __str__(self):
        return self.key