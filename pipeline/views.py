from django.shortcuts import render, redirect, reverse
from .forms import DatasetTrainForm, DatasetDescriptionForm, DatasetTestForm
from urllib.parse import urlencode
from .models import DatasetTrain, DatasetDescription, DatasetTest
import pandas as pd
import csv
from .pipe import pipe_train, pipe_predict


# Create your views here.
def upload_train(request):
    if request.POST:
        form = DatasetTrainForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.save()
            base_url = reverse('choose_columns')
            query_string = urlencode({'key': obj.key})
            url = '{}?{}'.format(base_url, query_string)
            return redirect(url)
        context = {'form': form}
        return render(request, 'train_dataset.html', context)
    else:
        form = DatasetTrainForm()
        context = {'form': form}
        return render(request, 'train_dataset.html', context)


def choose_columns(request):
    key = request.GET.get('key')
    print('key is', key)
    if request.POST:
        form = DatasetDescriptionForm(request.POST)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.key = key
            obj.save()
            base_url = reverse('train')
            query_string = urlencode({'key': obj.key})
            url = '{}?{}'.format(base_url, query_string)
            return redirect(url)
        context = {'form': form}
        return render(request, 'columns_choose.html', context)
    else:
        form = DatasetDescriptionForm()
        context = {'form': form}
        return render(request, 'columns_choose.html', context)


def train(request):
    key = request.GET.get('key')
    dataset_info = DatasetDescription.objects.get(key=key)
    dataset_train = DatasetTrain.objects.get(key=key)
    cat_columns = [i.strip() for i in dataset_info.categorical_columns.split(',')]
    print(cat_columns)
    print(type(cat_columns))
    num_columns = [i.strip() for i in dataset_info.numerical_columns.split(',')]
    print(num_columns)
    print(type(num_columns))
    id_column = dataset_info.id_column.strip()
    print(id_column)
    print(type(id_column))
    label_column = dataset_info.label_column.strip()
    print(label_column)
    print(type(label_column))
    df_train = pd.read_csv(dataset_train.upload)
    print(df_train)
    pipe_train(df_train, id_column, label_column, cat_columns, num_columns, key)
    return render(request, 'train.html', context={'message': 'We will email you, just kidding not now, maybe later;).'
                                                             ' Also please do not close this window,'
                                                             ' after loading will stop visit '
                                                             '127.0.0.1:8000/predict/?key={0}'.format(key)})


def predict(request):
    key = request.GET.get('key')
    if request.POST:
        form = DatasetTestForm(request.POST, request.FILES)
        if form.is_valid():
            obj = form.save(commit=False)
            obj.key = key
            obj.save()
            base_url = reverse('choose_columns_for_test')
            query_string = urlencode({'key': obj.key})
            url = '{}?{}'.format(base_url, query_string)
            return redirect(url)
        context = {'form': form}
        return render(request, 'test_dataset.html', context)
    else:
        form = DatasetTestForm()
        context = {'form': form}
        return render(request, 'test_dataset.html', context)


def choose_columns_for_test(request):
    key = request.GET.get('key')
    print('key is', key)
    dataset_info = DatasetDescription.objects.get(key=key)
    dataset_test = DatasetTest.objects.get(key=key)
    cat_columns = [i.strip() for i in dataset_info.categorical_columns.split(',')]
    num_columns = [i.strip() for i in dataset_info.numerical_columns.split(',')]
    id_column = dataset_info.id_column.strip()
    weights = 'files/{0}/weights'.format(key)
    df_features_test = pd.read_csv(dataset_test.upload)
    pred_df = pipe_predict(df_features_test, id_column, cat_columns, num_columns, weights, key)
    html_pred_table = pred_df.to_html()
    return render(request, 'test.html', context={'html_table': html_pred_table, 'possible_link': '127.0.0.1:8000/files/{0}/results/results.csv'.format(key) })
