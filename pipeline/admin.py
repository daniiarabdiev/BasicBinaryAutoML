from django.contrib import admin
from .models import DatasetTrain, DatasetDescription, Weight, ImportantFeature, DatasetTest


class DatasetTrainAdmin(admin.ModelAdmin):
    list_display = ['key']
    class Meta:
        model = DatasetTrain


admin.site.register(DatasetTrain, DatasetTrainAdmin)


class DatasetTestAdmin(admin.ModelAdmin):
    list_display = ['key']
    class Meta:
        model = DatasetTrain


admin.site.register(DatasetTest, DatasetTestAdmin)


class DatasetDescriptionAdmin(admin.ModelAdmin):
    list_display = ['key']
    class Meta:
        model = DatasetDescription


admin.site.register(DatasetDescription, DatasetDescriptionAdmin)


class WeightAdmin(admin.ModelAdmin):
    list_display = ['key']
    class Meta:
        model = DatasetDescription


admin.site.register(Weight, WeightAdmin)


class ImportantFeatureAdmin(admin.ModelAdmin):
    list_display = ['key']
    class Meta:
        model = DatasetDescription


admin.site.register(ImportantFeature, ImportantFeatureAdmin)