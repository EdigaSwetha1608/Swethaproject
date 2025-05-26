from django.shortcuts import render, redirect
from .models import News
from .forms import NewsForm
from .predict import manual_testing

def home(request):
    return render(request, 'detector/home.html')

def predict(request):
    if request.method == 'POST':
        form = NewsForm(request.POST)
        if form.is_valid():
            news_text = form.cleaned_data['text']
            news = News(text=news_text)
            news.save()
            return redirect('result', news_id=news.id)
    else:
        form = NewsForm()
    return render(request, 'detector/predict.html', {'form': form})

def result(request, news_id):
    news = News.objects.get(id=news_id)
    prediction = manual_testing(news.text)  # Call the updated manual_testing function
    news.is_fake = (prediction == "Fake News")  # Update the news object based on the prediction
    news.save()
    return render(request, 'detector/result.html', {'news': news, 'prediction': prediction})