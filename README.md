# denoiser
Noise removal with profile

## Структура класса Denoiser

Класс Denoiser предназначен для удаления шума из аудио-сигналов с использованием спектрального профиля шума. Он наследуется от torch.nn.Module и имеет следующие ключевые компоненты:

Атрибуты:
 - noise_profile: спектральный профиль шума, одномерный тензор размерности [freq_bins].
 - thr: пороговый уровень (threshold), гиперпараметр для регулирования степени подавления шума.
 - red_rate: коэффициент уменьшения шума (reduction rate), гиперпараметр.

Методы:
 - __init__(self, thr=1.0, red_rate=1.1): инициализация гиперпараметров.
 - fit(self, noise_sample): оценка спектрального профиля шума по входному аудио с шумом.
 - forward(self, audio_wav): удаление шума из входного аудио с использованием ранее оцененного профиля.

Причины выбора такой структуры

- Наследование от torch.nn.Module позволяет использовать встроенные методы и интегрироваться с другими компонентами PyTorch.
- Отдельные методы fit и forward обеспечивают модульность: сначала мы обучаем модель на чистом шуме, затем используем эту информацию для обработки нового аудио.
- Гиперпараметры thr и red_rate дают пользователю возможность тонко настраивать поведение денойзера.
