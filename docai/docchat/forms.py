from django import forms

class DocumentUploadForm(forms.Form):
    file = forms.FileField(label="Upload a PDF")

class QuestionForm(forms.Form):
    question = forms.CharField(label="Ask a question", max_length=500, widget=forms.TextInput(attrs={'size': '60'}))