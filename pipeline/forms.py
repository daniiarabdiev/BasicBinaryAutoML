from django import forms
from .models import DatasetTrain, DatasetTest,  DatasetDescription


class DatasetTrainForm(forms.ModelForm):
    class Meta:
        model = DatasetTrain
        fields = ['upload', 'key']


class DatasetTestForm(forms.ModelForm):
    class Meta:
        model = DatasetTest
        fields = ['upload']


class DatasetDescriptionForm(forms.ModelForm):
    class Meta:
        model = DatasetDescription
        fields = ['categorical_columns', 'numerical_columns', 'id_column', 'label_column']
