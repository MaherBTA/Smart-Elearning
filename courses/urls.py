from django.conf.urls import url
from . import views as course_views
from users import views as user_views

urlpatterns = [
    url(r'^$', course_views.courses, name='courses'),

    url(r'^student/(?P<course_name>[\w ]+)/$', user_views.course_homepage, name='course_homepage'),
    url(r'^student/(?P<course_name>[\w ]+)/(?P<slug>[\w-]+)/$', user_views.student_course,
        name='student_course'),
    url(r'^student/nex t/(?P<stype>[-\w.]+)/(?P<slevel>[-\w.]+)/(?P<ptype>[-\w.]+)/(?P<plevel>[-\w.]+)/(?P<success>[-\w.]+)/(?P<quiz_id>[-\d ]+)', user_views.evaluation_next,
        name='evaluation_next'),   

    url(r'^student/(?P<quiz_id>[\d ]+)', user_views.evaluation,
        name='evaluation'),     

    url(r'^professor/question/(?P<question_id>[\d ]+)/(?P<correct_answer_id>[\d ]+)', course_views.edit_question, name='edit_question'),   
               
    url(r'^professor/(?P<course_name>[\w ]+)/$', course_views.course, name='professor_course'),
    url(r'^professor/(?P<course_name>[\w ]+)/delete/$', course_views.delete_course, name='delete'),
    url(r'^professor/(?P<course_name>[\w ]+)/edit/$', course_views.update_course, name='edit'),

    url(r'^professor/(?P<course_name>[\w ]+)/students/$', course_views.list_students, name='list_students'),
    url(r'^professor/(?P<course_name>[\w ]+)/students/(?P<student_id>[\d ]+)/remove/$',
        course_views.remove_students, name='remove_students'),
    url(r'^professor/(?P<course_name>[\w ]+)/students/(?P<student_id>[\d ]+)/add/$',
        course_views.add_students, name='add_students'),

    url(r'^professor/(?P<course_name>[\w ]+)/(?P<slug>[\w-]+)/$', course_views.chapter, name='chapter'),
    url(r'^professor/edit/(?P<course_name>[\w ]+)/(?P<slug>[\w-]+)/$',
        course_views.update_chapter, name='edit_chapter'),
    url(r'^professor/delete/(?P<course_name>[\w ]+)/(?P<slug>[\w-]+)/$',
        course_views.delete_chapter, name='delete_chapter'),

    url(r'^professor/(?P<course_name>[\w ]+)/(?P<slug>[\w-]+)/(?P<txt_id>[\d ]+)/txt/edit/$',
        course_views.update_text_block, name='edit_txt'),
    url(r'^professor/txt/delete/(?P<txt_id>[\d ]+)/$', course_views.delete_text_block, name='delete_txt'),

    url(r'^professor/(?P<course_name>[\w ]+)/(?P<slug>[\w-]+)/(?P<yt_id>[\d ]+)/link/edit/$',
        course_views.update_yt_link, name='edit_link'),
    url(r'^professor/link/delete/(?P<yt_id>[\d ]+)/$', course_views.delete_yt_link, name='delete_link'),

    url(r'^professor/file/delete/(?P<file_id>[\d ]+)/$', course_views.delete_file, name='delete_file'),
    
    url(r'^professor/quiz/edit/(?P<quiz_id>[\w ]+)/(?P<question_id>[\w-]+)/(?P<answer>.*)/$', course_views.quiz, name='quiz'),
    
    url(r'^professor/quiz/delete/(?P<quiz_id>[\w ]+)/(?P<question_id>[\w-]+)/(?P<answer_id>[\w\-]+)/$', course_views.delete_answer, name='delete_answer'),

    url(r'^professor/quiz/delete/(?P<quiz_id>[\d ]+)/$', course_views.delete_quiz, name='delete_quiz'),
    
    url(r'^professor/quiz/generate/(?P<txt_block_id>[\d ]+)/(?P<slug>[\w-]+)/$', course_views.generate_quiz, name='generate_quiz'),

]
