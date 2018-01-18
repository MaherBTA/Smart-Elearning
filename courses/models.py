from __future__ import unicode_literals
import os

from django.utils.translation import gettext as _
from django.db import models
from users.models import UserProfile
from django.core.urlresolvers import reverse
from django.db.models.signals import pre_save
from django.utils.text import slugify
from django.dispatch import receiver
from django.template.defaultfilters import default


# Create your models here.
class Course(models.Model):
    course_name = models.CharField(unique=True, max_length=20)
    course_created_date = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(UserProfile, default=1)
    students = models.ManyToManyField(UserProfile, related_name='students_to_course')
    for_everybody = models.BooleanField(default=True)

    def __unicode__(self):
        return self.course_name


class Chapter(models.Model):
    chapter_name = models.CharField(max_length=20)
    chapter_created_date = models.DateTimeField(auto_now_add=True)
    course = models.ForeignKey(Course, on_delete=models.CASCADE, default=1)
    slug = models.SlugField(unique=True)

    def __unicode__(self):
        return self.chapter_name

    def get_absolute_url(self):
        return reverse("chapter", kwargs={"course_name": self.course,
                                          "slug": self.slug})

    def slug_default(self):
        slug = create_slug(new_slug=self.chapter_name)
        return slug


class Quiz(models.Model):
    
    chapter = models.ForeignKey(Chapter, on_delete=models.CASCADE, default=1)

    title = models.CharField(max_length=600,blank=False)

    description = models.TextField(blank=True)

    url = models.SlugField(max_length=60, blank=False)

    created_date = models.DateTimeField(auto_now_add=True)

    random_order = models.BooleanField(blank=False, default=False)

    success_text = models.TextField()

    fail_text = models.TextField()

    def __unicode__(self):
        return self.chapter_name

    def __str__(self):
        return self.title

    def get_questions(self):
        return self.question_set.all().select_subclasses()

    @property
    def get_max_score(self):
        return self.get_questions().count()

    def anon_score_id(self):
        return str(self.id) + "_score"

    def anon_q_list(self):
        return str(self.id) + "_q_list"

    def anon_q_data(self):
        return str(self.id) + "_data"


class Progress(models.Model):
    quiz = models.ForeignKey(Quiz, default=1)
    predicted_question = models.IntegerField(default=1)

class Steps(models.Model):
    progress = models.ForeignKey(Progress, default=1)
    order = models.IntegerField(default=1)
    question = models.IntegerField(default=1)    
    s_type = models.IntegerField(default=1)
    s_level = models.IntegerField(default=1)
    q_type = models.IntegerField(default=1)
    q_level = models.IntegerField(default=1)
    is_correct = models.IntegerField(default=1)


    
class Question(models.Model):
    """
    Base class for all question types.
    Shared properties placed here.
    """
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, default=1)

    figure = models.ImageField(upload_to='uploads/%Y/%m/%d',
                               blank=True,
                               null=True,
                               verbose_name=_("Figure"))
    
    level = models.IntegerField(default=1)
                        
    type = models.CharField(max_length=600, default="prob")
                               
    content = models.CharField(max_length=1000,
                               blank=False,
                               help_text=_("Enter the question text that "
                                           "you want displayed"),
                               verbose_name=_('Question'))

    correctAnswer = models.IntegerField(default=1,
                               verbose_name=_('Correct Answer'))

    explanation = models.TextField(max_length=2000,
                                   blank=True,
                                   help_text=_("Explanation to be shown "
                                               "after the question has "
                                               "been answered."),
                                   verbose_name=_('Explanation'))

class Answers(models.Model):
    """
    Base class for all Answers types.
    Shared properties placed here.
    """
    question = models.ForeignKey(Question, on_delete=models.CASCADE, default=1)
    answer = models.CharField(max_length=1000,
                               blank=False)  
    
def create_slug(instance=None, new_slug=None):
    slug = slugify(instance.chapter_name)

    if new_slug is not None:
        slug = new_slug

    qs = Chapter.objects.filter(slug=slug).order_by("-id")
    exists = qs.exists()

    if exists:
        new_slug = "%s-%s" % (slug, qs.first().id)
        return create_slug(instance, new_slug=new_slug)

    return slug


def pre_save_receiver(sender, instance, *args, **kwargs):
    if not instance.slug:
        instance.slug = create_slug(instance)

pre_save.connect(pre_save_receiver, sender=Chapter)


class TextBlock(models.Model):
    lesson = models.TextField()
    text_block_fk = models.ForeignKey(Chapter, default=1)
    date_created = models.DateTimeField(auto_now_add=True)


class YTLink(models.Model):
    link = models.URLField(max_length=200)
    yt_link_fk = models.ForeignKey(Chapter, default=1)
    date_created = models.DateTimeField(auto_now_add=True)


class FileUpload(models.Model):
    file = models.FileField(null=False, blank=False, default='')
    date_created = models.DateTimeField(auto_now_add=True)
    file_fk = models.ForeignKey(Chapter, default=1)


@receiver(models.signals.post_delete, sender=FileUpload)
def auto_delete_file_on_delete(sender, instance, **kwargs):
    if instance.file:
        if os.path.isfile(instance.file.path):
            os.remove(instance.file.path)
