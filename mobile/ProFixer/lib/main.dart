import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:url_launcher/url_launcher.dart';
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

void main() {
  runApp(const ProFixerApp());
}

enum AppLanguage { english, urdu, romanUrdu }

class AuthSessionStore {
  static const _tokenKey = 'auth_access_token';
  static const _userIdKey = 'auth_user_id';
  static const _nameKey = 'auth_user_name';
  static const _emailKey = 'auth_user_email';

  static String normalizeEmail(String email) => email.trim().toLowerCase();

  static Future<void> saveAuthPayload(Map<String, dynamic> payload) async {
    final user = Map<String, dynamic>.from(payload['user'] as Map? ?? {});
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenKey, '${payload['access_token'] ?? ''}');
    await prefs.setString(_userIdKey, '${user['id'] ?? ''}');
    await prefs.setString(_nameKey, '${user['name'] ?? ''}');
    await prefs.setString(_emailKey, normalizeEmail('${user['email'] ?? ''}'));
  }

  static Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_tokenKey);
    await prefs.remove(_userIdKey);
    await prefs.remove(_nameKey);
    await prefs.remove(_emailKey);
  }
}

// ── Design Tokens ──────────────────────────────────────────
class AppColors {
  static const primary      = Color(0xFF0A7C6E);
  static const primaryLight = Color(0xFF14B8A6);
  static const primaryDark  = Color(0xFF065F52);
  static const accent       = Color(0xFFFFB347);
  static const danger       = Color(0xFFEF4444);
  static const bg           = Color(0xFFF0F7F6);
  static const surface      = Colors.white;
  static const textDark     = Color(0xFF0F172A);
  static const textMid      = Color(0xFF475569);
  static const textLight    = Color(0xFF94A3B8);
  static const border       = Color(0xFFE2E8F0);
  static const cardShadow   = Color(0x1A0A7C6E);
}

class AppGradients {
  static const primary = LinearGradient(
    colors: [Color(0xFF065F52), Color(0xFF0D9488), Color(0xFF14B8A6)],
    begin: Alignment.topLeft, end: Alignment.bottomRight,
  );
  static const hero = LinearGradient(
    colors: [Color(0xFF042F2E), Color(0xFF065F52), Color(0xFF0F766E)],
    begin: Alignment.topLeft, end: Alignment.bottomRight,
  );
}

// ══════════════════════════════════════════════════════════
// FLOATING ANIMATED BACKGROUND
// ══════════════════════════════════════════════════════════
class _FloatingIcon {
  final IconData icon;
  final double x, y, size, speed, rotationSpeed, opacity;
  double angle;
  double rotAngle;

  _FloatingIcon({
    required this.icon,
    required this.x,
    required this.y,
    required this.size,
    required this.speed,
    required this.rotationSpeed,
    required this.opacity,
    required this.angle,
    required this.rotAngle,
  });
}

class AnimatedBackgroundIcons extends StatefulWidget {
  final Widget child;
  final bool darkMode;
  const AnimatedBackgroundIcons({Key? key, required this.child, this.darkMode = false}) : super(key: key);

  @override
  State<AnimatedBackgroundIcons> createState() => _AnimatedBackgroundIconsState();
}

class _AnimatedBackgroundIconsState extends State<AnimatedBackgroundIcons>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late List<_FloatingIcon> _icons;
  final _rand = math.Random(42);

  final List<IconData> _pool = [
    Icons.handyman_rounded,
    Icons.ac_unit_rounded,
    Icons.plumbing_rounded,
    Icons.electrical_services_rounded,
    Icons.build_rounded,
    Icons.home_repair_service_rounded,
    Icons.settings_rounded,
    Icons.bolt_rounded,
    Icons.water_drop_rounded,
    Icons.hvac_rounded,
  ];

  @override
  void initState() {
    super.initState();
    _icons = List.generate(18, (i) {
      return _FloatingIcon(
        icon: _pool[i % _pool.length],
        x: _rand.nextDouble(),
        y: _rand.nextDouble(),
        size: 18 + _rand.nextDouble() * 28,
        speed: 0.004 + _rand.nextDouble() * 0.008,
        rotationSpeed: (0.003 + _rand.nextDouble() * 0.006) * (_rand.nextBool() ? 1 : -1),
        opacity: 0.04 + _rand.nextDouble() * 0.09,
        angle: _rand.nextDouble() * math.pi * 2,
        rotAngle: _rand.nextDouble() * math.pi * 2,
      );
    });

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 1),
    )..repeat();

    _controller.addListener(() {
      setState(() {
        for (final ic in _icons) {
          ic.angle     += ic.speed;
          ic.rotAngle  += ic.rotationSpeed;
        }
      });
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final iconColor = widget.darkMode ? Colors.white : AppColors.primary;

    return Stack(
      children: [
        // Floating icons layer
        Positioned.fill(
          child: ClipRect(
            child: LayoutBuilder(builder: (context, constraints) {
              final w = constraints.maxWidth;
              final h = constraints.maxHeight;
              return Stack(
                children: _icons.map((ic) {
                  final cx = math.cos(ic.angle) * 0.18 * w;
                  final cy = math.sin(ic.angle * 0.7) * 0.14 * h;
                  final px = (ic.x * w) + cx;
                  final py = (ic.y * h) + cy;

                  return Positioned(
                    left: px - ic.size / 2,
                    top:  py - ic.size / 2,
                    child: Transform.rotate(
                      angle: ic.rotAngle,
                      child: Icon(
                        ic.icon,
                        size:  ic.size,
                        color: iconColor.withOpacity(ic.opacity),
                      ),
                    ),
                  );
                }).toList(),
              );
            }),
          ),
        ),
        // Actual page content on top
        widget.child,
      ],
    );
  }
}

// ══════════════════════════════════════════════════════════
// LANGUAGE + LOCALIZATION
// ══════════════════════════════════════════════════════════
class ProFixerApp extends StatefulWidget {
  const ProFixerApp({Key? key}) : super(key: key);
  @override State<ProFixerApp> createState() => _ProFixerAppState();
}

class _ProFixerAppState extends State<ProFixerApp> {
  AppLanguage _currentLang = AppLanguage.romanUrdu;
  void _changeLanguage(AppLanguage lang) => setState(() => _currentLang = lang);

  @override
  Widget build(BuildContext context) {
    return LanguageConfiguration(
      currentLanguage: _currentLang,
      onLanguageChanged: _changeLanguage,
      child: MaterialApp(
        title: 'Asan Khidmat Hub',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          primaryColor: AppColors.primary,
          scaffoldBackgroundColor: AppColors.bg,
          fontFamily: 'sans-serif',
          colorScheme: ColorScheme.fromSeed(seedColor: AppColors.primary),
        ),
        home: const AppNavigationWrapper(),
      ),
    );
  }
}

class LanguageConfiguration extends InheritedWidget {
  final AppLanguage currentLanguage;
  final ValueChanged<AppLanguage> onLanguageChanged;

  const LanguageConfiguration({
    Key? key,
    required this.currentLanguage,
    required this.onLanguageChanged,
    required Widget child,
  }) : super(key: key, child: child);

  static LanguageConfiguration? of(BuildContext context) =>
      context.dependOnInheritedWidgetOfExactType<LanguageConfiguration>();

  @override
  bool updateShouldNotify(LanguageConfiguration old) => old.currentLanguage != currentLanguage;
}

class LocalizedStrings {
  static String get(BuildContext context, String key) {
    final lang = LanguageConfiguration.of(context)?.currentLanguage ?? AppLanguage.romanUrdu;
    final Map<String, Map<AppLanguage, String>> v = {
      'appName':             { AppLanguage.english: 'Asan Khidmat Hub',       AppLanguage.urdu: 'آسان خدمت ہب',                                        AppLanguage.romanUrdu: 'Asan Khidmat Hub' },
      'tagline':             { AppLanguage.english: 'Convenience at home, in just one click', AppLanguage.urdu: 'گھر بیٹھے سہولت، ایک کلک پر',         AppLanguage.romanUrdu: 'Ghar bethe sahulat, aik click par' },
      'signUpTitle':         { AppLanguage.english: 'Create Account ✨',       AppLanguage.urdu: 'نیا اکاؤنٹ بنائیں ✨',                                 AppLanguage.romanUrdu: 'Naya Account Banayein ✨' },
      'signUpSub':           { AppLanguage.english: 'Register now to access premium informal services.', AppLanguage.urdu: 'آسان سہولیات کا فائدہ اٹھانے کے لیے ابھی رجسٹر کریں۔', AppLanguage.romanUrdu: 'Asan sahulat ka faida uthane ke liye register karein.' },
      'fullNameLabel':       { AppLanguage.english: 'Full Name',               AppLanguage.urdu: 'آپ کا نام (پورا نام)',                                 AppLanguage.romanUrdu: 'Aap ka Naam (Full Name)' },
      'emailLabel':          { AppLanguage.english: 'Email Address',           AppLanguage.urdu: 'ای میل ایڈریس',                                        AppLanguage.romanUrdu: 'Email Address' },
      'mobileLabel':         { AppLanguage.english: 'Mobile Number',           AppLanguage.urdu: 'موبائل نمبر',                                          AppLanguage.romanUrdu: 'Mobile Number' },
      'passwordLabel':       { AppLanguage.english: 'Password',                AppLanguage.urdu: 'پاس ورڈ',                                              AppLanguage.romanUrdu: 'Password' },
      'btnSignUp':           { AppLanguage.english: 'Create Account (Sign Up)',AppLanguage.urdu: 'اکاؤنٹ بنائیں',                                        AppLanguage.romanUrdu: 'Account Banayein (Sign Up)' },
      'alreadyAccount':      { AppLanguage.english: 'Already have an account? Log In', AppLanguage.urdu: 'پہلے سے اکاؤنٹ ہے؟ لاگ ان کریں',             AppLanguage.romanUrdu: 'Pehle se account hai? Log In karein' },
      'searchHint':          { AppLanguage.english: 'How can I help you today?', AppLanguage.urdu: 'میں آپ کی کیا مدد کر سکتا ہوں؟',                   AppLanguage.romanUrdu: 'Main kya madad karoon aap ki?' },
      'chatPrompt':          { AppLanguage.english: 'Tap below to chat with our AI Agent 💬', AppLanguage.urdu: 'اے آئی بوٹ سے بات کرنے کے لیے نیچے دبائیں 💬', AppLanguage.romanUrdu: 'Chatbot se baat karne ke liye niche 💬 dabayein' },
      'locationTitle':       { AppLanguage.english: 'Select Location',         AppLanguage.urdu: 'لوکیشن منتخب کریں',                                   AppLanguage.romanUrdu: 'Location Chunye' },
      'locationSub':         { AppLanguage.english: 'Select your city and operating area:', AppLanguage.urdu: 'اپنا شہر اور علاقہ منتخب کریں:',          AppLanguage.romanUrdu: 'Apna Sheher aur Ilaqa muntakhif karein:' },
      'cityLabel':           { AppLanguage.english: 'City',                    AppLanguage.urdu: 'شہر',                                                  AppLanguage.romanUrdu: 'Sheher (City)' },
      'areaLabel':           { AppLanguage.english: 'Area',                    AppLanguage.urdu: 'علاقہ',                                                AppLanguage.romanUrdu: 'Ilaqa (Area)' },
      'btnNext':             { AppLanguage.english: 'Proceed Forward',         AppLanguage.urdu: 'آگے چلیں',                                             AppLanguage.romanUrdu: 'Aagay Chalein' },
      'selectServiceTitle':  { AppLanguage.english: 'Select Required Service:',AppLanguage.urdu: 'سروس منتخب کریں:',                                     AppLanguage.romanUrdu: 'Service Muntakhif Karein:' },
      'availableNowTitle':   { AppLanguage.english: 'Available Now:',          AppLanguage.urdu: 'ابھی دستیاب ہے:',                                      AppLanguage.romanUrdu: 'Available Now (Abhi Maujood Hai):' },
      'busyTitle':           { AppLanguage.english: 'Currently Busy Providers:', AppLanguage.urdu: 'مصروف فراہم کنندگان:',                               AppLanguage.romanUrdu: 'Filhal Busy Providers:' },
      'selectServicePrompt': { AppLanguage.english: 'Please select a service from above categories.', AppLanguage.urdu: 'اوپر سے کوئی بھی ایک سروس منتخب کریں۔', AppLanguage.romanUrdu: 'Upar se koi bhi aik service select karein.' },
      'Karachi':        { AppLanguage.english: 'Karachi',        AppLanguage.urdu: 'کراچی',          AppLanguage.romanUrdu: 'Karachi' },
      'Lahore':         { AppLanguage.english: 'Lahore',         AppLanguage.urdu: 'لاہور',           AppLanguage.romanUrdu: 'Lahore' },
      'Islamabad':      { AppLanguage.english: 'Islamabad',      AppLanguage.urdu: 'اسلام آباد',      AppLanguage.romanUrdu: 'Islamabad' },
      'Saddar':         { AppLanguage.english: 'Saddar',         AppLanguage.urdu: 'صدر',             AppLanguage.romanUrdu: 'Saddar' },
      'Gulshan-e-Iqbal':{ AppLanguage.english: 'Gulshan-e-Iqbal',AppLanguage.urdu: 'گلشنِ اقبال',   AppLanguage.romanUrdu: 'Gulshan-e-Iqbal' },
      'Clifton':        { AppLanguage.english: 'Clifton',        AppLanguage.urdu: 'کلفٹن',           AppLanguage.romanUrdu: 'Clifton' },
      'Gulberg':        { AppLanguage.english: 'Gulberg',        AppLanguage.urdu: 'گلبرگ',           AppLanguage.romanUrdu: 'Gulberg' },
      'DHA Phase 5':    { AppLanguage.english: 'DHA Phase 5',    AppLanguage.urdu: 'ڈی ایچ اے فیز 5', AppLanguage.romanUrdu: 'DHA Phase 5' },
      'Johar Town':     { AppLanguage.english: 'Johar Town',     AppLanguage.urdu: 'جوہر ٹاؤن',       AppLanguage.romanUrdu: 'Johar Town' },
      'G-11':           { AppLanguage.english: 'G-11',           AppLanguage.urdu: 'جی الیون',        AppLanguage.romanUrdu: 'G-11' },
      'F-6':            { AppLanguage.english: 'F-6',            AppLanguage.urdu: 'ایف سکس',         AppLanguage.romanUrdu: 'F-6' },
      'I-9':            { AppLanguage.english: 'I-9',            AppLanguage.urdu: 'آئی نائن',        AppLanguage.romanUrdu: 'I-9' },
      'AC Repair':      { AppLanguage.english: 'AC Repair',      AppLanguage.urdu: 'اے سی کی مرمت',  AppLanguage.romanUrdu: 'AC Repair' },
      'Plumber':        { AppLanguage.english: 'Plumber',        AppLanguage.urdu: 'پلمبر (نل ساز)', AppLanguage.romanUrdu: 'Plumber' },
      'Electrician':    { AppLanguage.english: 'Electrician',    AppLanguage.urdu: 'الیکٹریشن',       AppLanguage.romanUrdu: 'Electrician' },
    };
    return v[key]?[lang] ?? key;
  }
}

// ══════════════════════════════════════════════════════════
// SHARED WIDGETS
// ══════════════════════════════════════════════════════════
class LanguageSwitcherRow extends StatelessWidget {
  const LanguageSwitcherRow({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final config  = LanguageConfiguration.of(context);
    final current = config?.currentLanguage ?? AppLanguage.romanUrdu;
    final isUrdu  = current == AppLanguage.urdu;

    final chips = Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        _chip(context, 'EN',    AppLanguage.english,   current == AppLanguage.english),
        const SizedBox(width: 5),
        _chip(context, 'اردو', AppLanguage.urdu,       current == AppLanguage.urdu),
        const SizedBox(width: 5),
        _chip(context, 'Roman', AppLanguage.romanUrdu, current == AppLanguage.romanUrdu),
      ],
    );

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.15),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: Colors.white.withOpacity(0.25)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: isUrdu ? [_logoutBtn(context), chips] : [chips, _logoutBtn(context)],
      ),
    );
  }

  Widget _chip(BuildContext context, String label, AppLanguage lang, bool selected) {
    return GestureDetector(
      onTap: () => LanguageConfiguration.of(context)?.onLanguageChanged(lang),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 220),
        padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 5),
        decoration: BoxDecoration(
          color: selected ? Colors.white.withOpacity(0.25) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
          border: selected ? Border.all(color: Colors.white.withOpacity(0.5)) : null,
        ),
        child: Text(label,
          style: TextStyle(
            fontSize: 11, fontWeight: FontWeight.w700, letterSpacing: 0.3,
            color: selected ? Colors.white : Colors.white.withOpacity(0.65),
          )),
      ),
    );
  }

  Widget _logoutBtn(BuildContext context) {
    return GestureDetector(
      onTap: () {
        AuthSessionStore.logout();
        BackendClient.instance.logout();
        Navigator.of(context).pushAndRemoveUntil(
          MaterialPageRoute(builder: (_) => const LoginPage()), (r) => false);
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
        decoration: BoxDecoration(
          color: Colors.red.withOpacity(0.2),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.red.withOpacity(0.4)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: const [
          Icon(Icons.logout_rounded, size: 12, color: Colors.redAccent),
          SizedBox(width: 5),
          Text('Logout', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: Colors.redAccent)),
        ]),
      ),
    );
  }
}

class PrimaryButton extends StatelessWidget {
  final String label;
  final VoidCallback onPressed;
  final IconData? icon;
  const PrimaryButton({Key? key, required this.label, required this.onPressed, this.icon}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity, height: 54,
      decoration: BoxDecoration(
        gradient: AppGradients.primary,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [BoxShadow(color: AppColors.primary.withOpacity(0.40), blurRadius: 18, offset: const Offset(0, 7))],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onPressed,
          child: Center(child: Row(mainAxisSize: MainAxisSize.min, children: [
            if (icon != null) ...[Icon(icon, color: Colors.white, size: 18), const SizedBox(width: 8)],
            Text(label, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w800, color: Colors.white, letterSpacing: 0.4)),
          ])),
        ),
      ),
    );
  }
}

class StyledField extends StatelessWidget {
  final TextEditingController controller;
  final String label, hint;
  final IconData icon;
  final bool isPassword, passwordHidden, isUrdu;
  final VoidCallback? onTogglePassword;
  final TextInputType keyboardType;
  final List<TextInputFormatter>? inputFormatters;
  final String? Function(String?)? validator;
  final void Function(String)? onChanged;

  const StyledField({
    Key? key,
    required this.controller, required this.label, required this.hint,
    required this.icon, required this.isUrdu,
    this.isPassword = false, this.passwordHidden = true,
    this.onTogglePassword, this.keyboardType = TextInputType.text,
    this.inputFormatters, this.validator, this.onChanged,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: isUrdu ? CrossAxisAlignment.end : CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 7, left: 2, right: 2),
          child: Text(label, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: AppColors.textMid, letterSpacing: 0.5)),
        ),
        Container(
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppColors.border),
            boxShadow: [BoxShadow(color: AppColors.cardShadow, blurRadius: 8, offset: const Offset(0, 2))],
          ),
          child: TextFormField(
            controller: controller,
            keyboardType: keyboardType,
            inputFormatters: inputFormatters,
            obscureText: isPassword && passwordHidden,
            textAlign: isUrdu ? TextAlign.right : TextAlign.left,
            onChanged: onChanged,
            validator: validator,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, color: AppColors.textDark),
            decoration: InputDecoration(
              prefixIcon: isUrdu ? null : Icon(icon, color: AppColors.primary, size: 20),
              suffixIcon: isUrdu
                  ? (isPassword
                      ? IconButton(icon: Icon(passwordHidden ? Icons.visibility_off_rounded : Icons.visibility_rounded, color: AppColors.textLight, size: 20), onPressed: onTogglePassword)
                      : Icon(icon, color: AppColors.primary, size: 20))
                  : (isPassword
                      ? IconButton(icon: Icon(passwordHidden ? Icons.visibility_off_rounded : Icons.visibility_rounded, color: AppColors.textLight, size: 20), onPressed: onTogglePassword)
                      : null),
              hintText: hint,
              hintStyle: const TextStyle(color: AppColors.textLight, fontSize: 13),
              border: InputBorder.none,
              contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
            ),
          ),
        ),
      ],
    );
  }
}

// ══════════════════════════════════════════════════════════
// PROVIDER MODEL
// ══════════════════════════════════════════════════════════
class ProviderModel {
  final int? providerId;
  final String name, specialty, nextAvailableIn, phone, category, city, area, priceRange, availability;
  final double rating;
  final int reviews, eta;
  final bool isAvailable;
  ProviderModel({
    this.providerId,
    required this.name,
    required this.specialty,
    required this.rating,
    required this.reviews,
    required this.isAvailable,
    required this.eta,
    required this.nextAvailableIn,
    required this.phone,
    this.category = '',
    this.city = '',
    this.area = '',
    this.priceRange = '',
    this.availability = '',
  });

  factory ProviderModel.fromJson(Map<String, dynamic> json) {
    final availability = '${json['availability'] ?? ''}';
    final isAvailable = availability.toLowerCase() == 'available';
    final responseMinutes = _asInt(json['response_time_minutes']);
    final category = '${json['category'] ?? ''}';
    final city = '${json['city'] ?? ''}';
    final area = '${json['area'] ?? ''}';
    return ProviderModel(
      providerId: _asIntOrNull(json['provider_id']),
      name: '${json['provider_name'] ?? 'Unknown Provider'}',
      specialty: category.isEmpty ? 'Service Provider' : category,
      rating: _asDouble(json['rating']),
      reviews: _asInt(json['completed_jobs']),
      isAvailable: isAvailable,
      eta: responseMinutes,
      nextAvailableIn: isAvailable ? '' : (responseMinutes > 0 ? '$responseMinutes mins' : 'later'),
      phone: '${json['phone_number'] ?? ''}',
      category: category,
      city: city,
      area: area,
      priceRange: '${json['price_range'] ?? ''}',
      availability: availability,
    );
  }
}

class ChatResult {
  final String response;
  final String status;
  final String sessionId;
  final List<dynamic> providers;
  final Map<String, dynamic>? booking;
  final List<dynamic> handoffTrace;
  final List<dynamic> workflowTrace;

  ChatResult({
    required this.response,
    required this.status,
    required this.sessionId,
    required this.providers,
    required this.booking,
    this.handoffTrace = const [],
    this.workflowTrace = const [],
  });

  /// Whether the backend is asking the user to confirm a booking.
  bool get isAwaitingBookingConfirmation => status == 'awaiting_booking_confirmation';
  bool get isBookingConfirmed => status == 'booking_confirmed';
  bool get isBookingCancelled => status == 'booking_cancelled';

  factory ChatResult.fromJson(Map<String, dynamic> json) {
    return ChatResult(
      response: '${json['response'] ?? ''}',
      status: '${json['status'] ?? 'unknown'}',
      sessionId: '${json['session_id'] ?? ''}',
      providers: json['providers'] is List ? json['providers'] as List<dynamic> : const [],
      booking: json['booking'] is Map ? Map<String, dynamic>.from(json['booking'] as Map) : null,
      handoffTrace: json['handoff_trace'] is List ? json['handoff_trace'] as List<dynamic> : const [],
      workflowTrace: json['workflow_trace'] is List ? json['workflow_trace'] as List<dynamic> : const [],
    );
  }
}

class BookingModel {
  final String bookingId;
  final String sessionId;
  final int providerId;
  final String serviceType;
  final String location;
  final String status;
  final String? scheduledTime;
  final String? userNotes;
  final String createdAt;

  BookingModel({
    required this.bookingId,
    required this.sessionId,
    required this.providerId,
    required this.serviceType,
    required this.location,
    required this.status,
    this.scheduledTime,
    this.userNotes,
    required this.createdAt,
  });

  factory BookingModel.fromJson(Map<String, dynamic> json) {
    return BookingModel(
      bookingId: '${json['booking_id'] ?? json['id'] ?? ''}',
      sessionId: '${json['session_id'] ?? ''}',
      providerId: _asInt(json['provider_id']),
      serviceType: '${json['service_type'] ?? ''}',
      location: '${json['location'] ?? ''}',
      status: '${json['status'] ?? 'PENDING'}',
      scheduledTime: json['scheduled_time']?.toString(),
      userNotes: json['user_notes']?.toString(),
      createdAt: '${json['created_at'] ?? ''}',
    );
  }
}

// ── Production API Configuration ───────────────────────────
class ApiConfig {
  static const Duration connectionTimeout = Duration(seconds: 20);
  static const Duration readTimeout = Duration(seconds: 45);
  static const Duration chatTimeout = Duration(seconds: 90);
  static const Duration wsConnectTimeout = Duration(seconds: 8);
  static const Duration wsResponseTimeout = Duration(seconds: 90);
  static const int maxWsRetries = 2;
  static const Duration wsRetryDelay = Duration(seconds: 1);
}

class BackendClient {
  BackendClient._() {
    _http.connectionTimeout = ApiConfig.connectionTimeout;
  }

  static final BackendClient instance = BackendClient._();

  // Production URL: override at build time with --dart-define=PROFIXER_API_BASE_URL=...
  static const String apiBaseUrl = String.fromEnvironment(
    'PROFIXER_API_BASE_URL',
    defaultValue: 'http://20.17.177.214:8000',
  );
  static const String configuredWsBaseUrl = String.fromEnvironment(
    'PROFIXER_WS_BASE_URL',
    defaultValue: '',
  );

  final HttpClient _http = HttpClient();
  String? _sessionId;

  Future<Map<String, dynamic>> signup({
    required String name,
    required String email,
    required String password,
  }) {
    return _postJson('/api/v1/auth/signup', {
      'name': name,
      'email': email,
      'password': password,
    });
  }

  Future<Map<String, dynamic>> login({
    required String email,
    required String password,
  }) {
    return _postJson('/api/v1/auth/login', {
      'email': email,
      'password': password,
    });
  }

  Future<void> logout() async {
    try {
      await _postJson('/api/v1/auth/logout', {});
    } catch (_) {
      // Logout is client-side for JWT; backend call is best-effort.
    }
  }

  /// Returns current session ID, creating one from backend if needed.
  Future<String> ensureSession() async {
    if (_sessionId != null && _sessionId!.isNotEmpty) return _sessionId!;
    final payload = await _postJson('/api/v1/sessions', {});
    _sessionId = '${payload['session_id'] ?? ''}';
    return _sessionId!;
  }

  Future<List<ProviderModel>> fetchProviders({
    required String query,
    required String city,
    required String area,
  }) async {
    final payload = await _getJson('/api/v1/providers', {
      'query': query,
      'city': city,
      'area': area,
      'limit': '10',
      'sort_by': 'score',
    });
    final rawProviders = payload['providers'];
    if (rawProviders is! List) return [];
    return rawProviders
        .whereType<Map>()
        .map((item) => ProviderModel.fromJson(Map<String, dynamic>.from(item)))
        .toList();
  }

  /// Fetch all cities with areas and provider counts from discovery endpoint.
  Future<List<Map<String, dynamic>>> fetchCities() async {
    final payload = await _getJson('/api/v1/discovery/cities', {});
    final raw = payload['cities'];
    if (raw is! List) return [];
    return raw.whereType<Map>().map((c) => Map<String, dynamic>.from(c)).toList();
  }

  /// Fetch all service categories with icons and provider counts.
  Future<List<Map<String, dynamic>>> fetchCategories() async {
    final payload = await _getJson('/api/v1/discovery/categories', {});
    final raw = payload['categories'];
    if (raw is! List) return [];
    return raw.whereType<Map>().map((c) => Map<String, dynamic>.from(c)).toList();
  }

  /// Fetch areas for a specific city.
  Future<List<String>> fetchCityAreas(String city) async {
    try {
      final payload = await _getJson('/api/v1/discovery/cities/$city/areas', {});
      final raw = payload['areas'];
      if (raw is! List) return [];
      // areas endpoint returns objects with 'name' and 'provider_count'
      return raw.whereType<Map>().map((a) => '${a['name'] ?? a}').toList();
    } catch (_) {
      return [];
    }
  }

  /// Check backend health.
  Future<Map<String, dynamic>> checkHealth() async {
    return _getJson('/api/v1/health', {});
  }

  /// Create a new session explicitly.
  Future<String> createSession() async {
    final payload = await _postJson('/api/v1/sessions', {});
    _sessionId = '${payload['session_id'] ?? ''}';
    return _sessionId!;
  }

  /// Fetch full provider detail by ID.
  Future<Map<String, dynamic>> fetchProviderDetail(int providerId) async {
    return _getJson('/api/v1/providers/$providerId', {});
  }

  /// Create a booking via the backend.
  Future<BookingModel> createBooking({
    required String sessionId,
    required int providerId,
    required String serviceType,
    required String location,
    String? scheduledTime,
    String? userNotes,
  }) async {
    final body = <String, dynamic>{
      'session_id': sessionId,
      'provider_id': providerId,
      'service_type': serviceType,
      'location': location,
    };
    if (scheduledTime != null) body['scheduled_time'] = scheduledTime;
    if (userNotes != null) body['user_notes'] = userNotes;
    final payload = await _postJson('/api/v1/bookings', body);
    return BookingModel.fromJson(payload);
  }

  /// Send chat message — tries WebSocket with retry, then REST fallback.
  Future<ChatResult> sendChat(String message) async {
    // Ensure we have a session before chatting.
    await ensureSession();
    for (int attempt = 0; attempt <= ApiConfig.maxWsRetries; attempt++) {
      try {
        return await _sendChatOverWebSocket(message);
      } catch (_) {
        if (attempt < ApiConfig.maxWsRetries) {
          await Future.delayed(ApiConfig.wsRetryDelay);
        }
      }
    }
    // All WS attempts failed — fall back to REST.
    return _sendChatOverRest(message);
  }

  Future<ChatResult> _sendChatOverRest(String message) async {
    final payload = await _postJson('/api/v1/chat', {
      'session_id': _sessionId,
      'message': message,
    });
    final result = ChatResult.fromJson(payload);
    if (result.sessionId.isNotEmpty) _sessionId = result.sessionId;
    return result;
  }

  Future<ChatResult> _sendChatOverWebSocket(String message) async {
    WebSocket? socket;
    try {
      socket = await WebSocket.connect(_wsUri('/api/v1/ws/chat', {
        'session_id': _sessionId,
      }).toString()).timeout(ApiConfig.wsConnectTimeout);

      socket.add(jsonEncode({
        'session_id': _sessionId,
        'message': message,
      }));

      await for (final raw in socket.timeout(ApiConfig.wsResponseTimeout)) {
        final payload = jsonDecode('$raw');
        if (payload is! Map) continue;
        final data = Map<String, dynamic>.from(payload);
        if (data['type'] == 'session') {
          final sessionId = '${data['session_id'] ?? ''}';
          if (sessionId.isNotEmpty) _sessionId = sessionId;
          continue;
        }
        if (data['type'] == 'chat_response') {
          final result = ChatResult.fromJson(data);
          if (result.sessionId.isNotEmpty) _sessionId = result.sessionId;
          return result;
        }
        if (data['type'] == 'error') {
          throw StateError('${data['message'] ?? 'Backend error'}');
        }
      }
      throw StateError('No chat response received');
    } finally {
      await socket?.close();
    }
  }

  Future<Map<String, dynamic>> _getJson(String path, Map<String, String?> query) async {
    final request = await _http.getUrl(_apiUri(path, query));
    request.headers.set(HttpHeaders.acceptHeader, ContentType.json.mimeType);
    return _readJsonResponse(await request.close().timeout(ApiConfig.readTimeout));
  }

  Future<Map<String, dynamic>> _postJson(String path, Map<String, dynamic> body) async {
    final request = await _http.postUrl(_apiUri(path));
    request.headers.contentType = ContentType.json;
    request.headers.set(HttpHeaders.acceptHeader, ContentType.json.mimeType);
    request.write(jsonEncode(body));
    return _readJsonResponse(await request.close().timeout(ApiConfig.chatTimeout));
  }

  Future<Map<String, dynamic>> _readJsonResponse(HttpClientResponse response) async {
    final text = await utf8.decoder.bind(response).join();
    if (response.statusCode < 200 || response.statusCode >= 300) {
      // Parse FastAPI error detail if available.
      String errorMsg = 'Server error (${response.statusCode})';
      try {
        final errBody = jsonDecode(text);
        if (errBody is Map && errBody['detail'] != null) {
          errorMsg = '${errBody['detail']}';
        }
      } catch (_) {
        // Use generic message.
      }
      if (response.statusCode == 408) errorMsg = 'Request timed out. Please try again.';
      if (response.statusCode == 503) errorMsg = 'Service temporarily unavailable. Please wait.';
      throw StateError(errorMsg);
    }
    final decoded = jsonDecode(text);
    if (decoded is Map) {
      return Map<String, dynamic>.from(decoded);
    }
    throw StateError('Backend returned an unexpected response');
  }

  Uri _apiUri(String path, [Map<String, String?> query = const {}]) {
    final base = Uri.parse(apiBaseUrl);
    final cleanedQuery = <String, String>{};
    query.forEach((key, value) {
      if (value != null && value.trim().isNotEmpty) cleanedQuery[key] = value;
    });
    return base.replace(
      path: _joinPath(base.path, path),
      queryParameters: cleanedQuery.isEmpty ? null : cleanedQuery,
    );
  }

  Uri _wsUri(String path, [Map<String, String?> query = const {}]) {
    final wsBase = configuredWsBaseUrl.isNotEmpty ? configuredWsBaseUrl : _deriveWsBaseUrl();
    final base = Uri.parse(wsBase);
    final cleanedQuery = <String, String>{};
    query.forEach((key, value) {
      if (value != null && value.trim().isNotEmpty) cleanedQuery[key] = value;
    });
    return base.replace(
      path: _joinPath(base.path, path),
      queryParameters: cleanedQuery.isEmpty ? null : cleanedQuery,
    );
  }

  String _deriveWsBaseUrl() {
    if (apiBaseUrl.startsWith('https://')) return 'wss://${apiBaseUrl.substring(8)}';
    if (apiBaseUrl.startsWith('http://')) return 'ws://${apiBaseUrl.substring(7)}';
    return apiBaseUrl;
  }

  String _joinPath(String basePath, String path) {
    final left = basePath.endsWith('/') ? basePath.substring(0, basePath.length - 1) : basePath;
    final right = path.startsWith('/') ? path.substring(1) : path;
    if (left.isEmpty) return right;
    if (right.isEmpty) return left;
    return '$left/$right';
  }
}

int _asInt(dynamic value) => _asIntOrNull(value) ?? 0;

int? _asIntOrNull(dynamic value) {
  if (value is int) return value;
  if (value is num) return value.toInt();
  return int.tryParse('$value');
}

double _asDouble(dynamic value) {
  if (value is num) return value.toDouble();
  return double.tryParse('$value') ?? 0.0;
}

String _cleanBackendError(Object error) {
  return error.toString().replaceFirst('Bad state: ', '');
}

// ══════════════════════════════════════════════════════════
// GLOBAL WRAPPER
// ══════════════════════════════════════════════════════════
class AppNavigationWrapper extends StatefulWidget {
  const AppNavigationWrapper({Key? key}) : super(key: key);
  @override State<AppNavigationWrapper> createState() => _AppNavigationWrapperState();
}

class _AppNavigationWrapperState extends State<AppNavigationWrapper> {
  final GlobalKey<NavigatorState> _navKey = GlobalKey<NavigatorState>();

  @override
  Widget build(BuildContext context) {
    final isUrdu = LanguageConfiguration.of(context)?.currentLanguage == AppLanguage.urdu;
    return Scaffold(
      floatingActionButtonLocation: isUrdu ? FloatingActionButtonLocation.startFloat : FloatingActionButtonLocation.endFloat,
      floatingActionButton: Container(
        decoration: BoxDecoration(
          gradient: AppGradients.primary, shape: BoxShape.circle,
          boxShadow: [BoxShadow(color: AppColors.primary.withOpacity(0.45), blurRadius: 20, offset: const Offset(0, 6))],
        ),
        child: FloatingActionButton(
          backgroundColor: Colors.transparent, elevation: 0,
          child: const Icon(Icons.chat_bubble_rounded, color: Colors.white, size: 26),
          onPressed: () => showModalBottomSheet(
            context: context, isScrollControlled: true, backgroundColor: Colors.transparent,
            builder: (_) => const ChatBotWidget(),
          ),
        ),
      ),
      body: Navigator(
        key: _navKey,
        onGenerateRoute: (_) => MaterialPageRoute(builder: (_) => const LoginPage()),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════
// PAGE: CREATE ACCOUNT
// ══════════════════════════════════════════════════════════
class LoginPage extends StatefulWidget {
  const LoginPage({Key? key}) : super(key: key);
  @override State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailCtrl = TextEditingController();
  final _passwordCtrl = TextEditingController();
  bool _pwdHidden = true;

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passwordCtrl.dispose();
    super.dispose();
  }

  void _showAuthError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: AppColors.danger),
    );
  }

  Future<void> _demoLogin() async {
    if (!_formKey.currentState!.validate()) return;
    try {
      final payload = await BackendClient.instance.login(
        email: AuthSessionStore.normalizeEmail(_emailCtrl.text),
        password: _passwordCtrl.text,
      );
      await AuthSessionStore.saveAuthPayload(payload);
      if (!mounted) return;
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const WelcomeHomePage()));
    } catch (error) {
      if (!mounted) return;
      _showAuthError(_cleanBackendError(error));
    }
  }

  @override
  Widget build(BuildContext context) {
    final isUrdu = LanguageConfiguration.of(context)?.currentLanguage == AppLanguage.urdu;
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: AnimatedBackgroundIcons(
        child: SafeArea(
          child: GestureDetector(
            onTap: () => FocusScope.of(context).unfocus(),
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              child: Column(children: [
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.fromLTRB(24, 20, 24, 34),
                  decoration: const BoxDecoration(
                    gradient: AppGradients.hero,
                    borderRadius: BorderRadius.vertical(bottom: Radius.circular(36)),
                  ),
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    const LanguageSwitcherRow(),
                    const SizedBox(height: 28),
                    Row(children: [
                      Container(
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(color: Colors.white.withOpacity(0.15), borderRadius: BorderRadius.circular(14)),
                        child: const Icon(Icons.handyman_rounded, color: Colors.white, size: 24),
                      ),
                      const SizedBox(width: 12),
                      Text(LocalizedStrings.get(context, 'appName'),
                        style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w900, color: Colors.white)),
                    ]),
                    const SizedBox(height: 22),
                    const Text('Welcome back',
                      style: TextStyle(fontSize: 30, fontWeight: FontWeight.w900, color: Colors.white, height: 1.2)),
                    const SizedBox(height: 8),
                    Text('Login to continue booking trusted local services.',
                      style: TextStyle(fontSize: 13, color: Colors.white.withOpacity(0.75), height: 1.5)),
                  ]),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 26, 20, 30),
                  child: Form(
                    key: _formKey,
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      StyledField(
                        controller: _emailCtrl,
                        label: 'Email Address',
                        hint: 'name@example.com',
                        icon: Icons.mail_outline_rounded,
                        isUrdu: isUrdu,
                        keyboardType: TextInputType.emailAddress,
                        validator: (v) {
                          if (v == null || v.trim().isEmpty) return 'Required';
                          if (!RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$').hasMatch(v.trim())) return 'Invalid email';
                          return null;
                        },
                      ),
                      const SizedBox(height: 16),
                      StyledField(
                        controller: _passwordCtrl,
                        label: 'Password',
                        hint: 'Password',
                        icon: Icons.lock_outline_rounded,
                        isUrdu: isUrdu,
                        isPassword: true,
                        passwordHidden: _pwdHidden,
                        onTogglePassword: () => setState(() => _pwdHidden = !_pwdHidden),
                        validator: (v) => (v == null || v.length < 8) ? 'Minimum 8 characters' : null,
                      ),
                      const SizedBox(height: 30),
                      PrimaryButton(label: 'Login', icon: Icons.login_rounded, onPressed: _demoLogin),
                      const SizedBox(height: 18),
                      Center(
                        child: GestureDetector(
                          onTap: () => Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const CreateAccountPage())),
                          child: const Text(
                            'New here? Create an account',
                            style: TextStyle(color: AppColors.primary, fontWeight: FontWeight.w800, fontSize: 13),
                          ),
                        ),
                      ),
                    ]),
                  ),
                ),
              ]),
            ),
          ),
        ),
      ),
    );
  }
}

class CreateAccountPage extends StatefulWidget {
  const CreateAccountPage({Key? key}) : super(key: key);
  @override State<CreateAccountPage> createState() => _CreateAccountPageState();
}

class _CreateAccountPageState extends State<CreateAccountPage> {
  final _formKey      = GlobalKey<FormState>();
  final _nameCtrl     = TextEditingController();
  final _emailCtrl    = TextEditingController();
  final _phoneCtrl    = TextEditingController();
  final _passwordCtrl = TextEditingController();
  bool   _pwdHidden       = true;
  String _strengthText    = '';
  Color  _strengthColor   = Colors.transparent;
  double _strengthProgress = 0.0;

  void _checkStrength(String pw) {
    if (pw.isEmpty) { setState(() { _strengthText = ''; _strengthColor = Colors.transparent; _strengthProgress = 0; }); return; }
    int s = 0;
    if (pw.length >= 8) s++;
    if (pw.contains(RegExp(r'[A-Z]')) && pw.contains(RegExp(r'[a-z]'))) s++;
    if (pw.contains(RegExp(r'[0-9]'))) s++;
    if (pw.contains(RegExp(r'[!@#$%^&*(),.?":{}|<>]'))) s++;
    setState(() {
      if (s <= 2)       { _strengthText = 'Weak 🔴';   _strengthColor = Colors.red;    _strengthProgress = 0.33; }
      else if (s == 3)  { _strengthText = 'Medium 🟡'; _strengthColor = Colors.orange; _strengthProgress = 0.66; }
      else              { _strengthText = 'Strong 🟢';  _strengthColor = Colors.green;  _strengthProgress = 1.0; }
    });
  }

  Future<void> _demoSignup() async {
    if (!_formKey.currentState!.validate()) return;
    try {
      final payload = await BackendClient.instance.signup(
        name: _nameCtrl.text.trim(),
        email: AuthSessionStore.normalizeEmail(_emailCtrl.text),
        password: _passwordCtrl.text,
      );
      await AuthSessionStore.saveAuthPayload(payload);
      if (!mounted) return;
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const WelcomeHomePage()));
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(_cleanBackendError(error)), backgroundColor: AppColors.danger),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final isUrdu = LanguageConfiguration.of(context)?.currentLanguage == AppLanguage.urdu;
    final lang   = LanguageConfiguration.of(context)?.currentLanguage ?? AppLanguage.romanUrdu;

    return Scaffold(
      backgroundColor: AppColors.bg,
      body: AnimatedBackgroundIcons(
        darkMode: false,
        child: SafeArea(
          child: GestureDetector(
            onTap: () => FocusScope.of(context).unfocus(),
            child: SingleChildScrollView(
              physics: const BouncingScrollPhysics(),
              child: Column(
                children: [
                  // Hero
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.fromLTRB(24, 20, 24, 32),
                    decoration: const BoxDecoration(
                      gradient: AppGradients.hero,
                      borderRadius: BorderRadius.vertical(bottom: Radius.circular(36)),
                    ),
                    child: Column(
                      crossAxisAlignment: isUrdu ? CrossAxisAlignment.end : CrossAxisAlignment.start,
                      children: [
                        const LanguageSwitcherRow(),
                        const SizedBox(height: 28),
                        Row(
                          mainAxisAlignment: isUrdu ? MainAxisAlignment.end : MainAxisAlignment.start,
                          children: [
                            if (!isUrdu) ...[
                              Container(
                                padding: const EdgeInsets.all(10),
                                decoration: BoxDecoration(color: Colors.white.withOpacity(0.15), borderRadius: BorderRadius.circular(14)),
                                child: const Icon(Icons.handyman_rounded, color: Colors.white, size: 24),
                              ),
                              const SizedBox(width: 12),
                            ],
                            Text(LocalizedStrings.get(context, 'appName'),
                              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w900, color: Colors.white, letterSpacing: 0.3)),
                            if (isUrdu) ...[
                              const SizedBox(width: 12),
                              Container(
                                padding: const EdgeInsets.all(10),
                                decoration: BoxDecoration(color: Colors.white.withOpacity(0.15), borderRadius: BorderRadius.circular(14)),
                                child: const Icon(Icons.handyman_rounded, color: Colors.white, size: 24),
                              ),
                            ],
                          ],
                        ),
                        const SizedBox(height: 20),
                        Text(LocalizedStrings.get(context, 'signUpTitle'),
                          textAlign: isUrdu ? TextAlign.right : TextAlign.left,
                          style: const TextStyle(fontSize: 30, fontWeight: FontWeight.w900, color: Colors.white, height: 1.2)),
                        const SizedBox(height: 8),
                        Text(LocalizedStrings.get(context, 'signUpSub'),
                          textAlign: isUrdu ? TextAlign.right : TextAlign.left,
                          style: TextStyle(fontSize: 13, color: Colors.white.withOpacity(0.75), height: 1.5)),
                      ],
                    ),
                  ),

                  // Form
                  Padding(
                    padding: const EdgeInsets.fromLTRB(20, 24, 20, 30),
                    child: Form(
                      key: _formKey,
                      child: Column(
                        crossAxisAlignment: isUrdu ? CrossAxisAlignment.end : CrossAxisAlignment.start,
                        children: [
                          StyledField(controller: _nameCtrl, label: LocalizedStrings.get(context, 'fullNameLabel'), hint: 'Anum Ejaz', icon: Icons.person_outline_rounded, isUrdu: isUrdu, validator: (v) => (v == null || v.trim().isEmpty) ? 'Required' : null),
                          const SizedBox(height: 16),
                          StyledField(controller: _emailCtrl, label: LocalizedStrings.get(context, 'emailLabel'), hint: 'name@example.com', icon: Icons.mail_outline_rounded, isUrdu: isUrdu, keyboardType: TextInputType.emailAddress,
                            validator: (v) { if (v == null || v.trim().isEmpty) return 'Required'; if (!RegExp(r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$').hasMatch(v.trim())) return 'Invalid email'; return null; }),
                          const SizedBox(height: 16),
                          StyledField(controller: _phoneCtrl, label: LocalizedStrings.get(context, 'mobileLabel'), hint: '03001234567', icon: Icons.phone_android_rounded, isUrdu: isUrdu, keyboardType: TextInputType.phone,
                            inputFormatters: [FilteringTextInputFormatter.digitsOnly, LengthLimitingTextInputFormatter(11)],
                            validator: (v) { if (v == null || v.isEmpty) return null; if (v.length < 11) return 'Must be 11 digits'; return null; }),
                          const SizedBox(height: 16),
                          StyledField(controller: _passwordCtrl, label: LocalizedStrings.get(context, 'passwordLabel'), hint: '••••••••', icon: Icons.lock_outline_rounded, isUrdu: isUrdu,
                            isPassword: true, passwordHidden: _pwdHidden, onTogglePassword: () => setState(() => _pwdHidden = !_pwdHidden),
                            onChanged: _checkStrength, validator: (v) => (v == null || v.length < 8) ? 'Minimum 8 characters' : null),

                          if (_passwordCtrl.text.isNotEmpty) ...[
                            const SizedBox(height: 10),
                            ClipRRect(borderRadius: BorderRadius.circular(6),
                              child: LinearProgressIndicator(value: _strengthProgress, backgroundColor: AppColors.border, valueColor: AlwaysStoppedAnimation<Color>(_strengthColor), minHeight: 5)),
                            const SizedBox(height: 5),
                            Align(alignment: isUrdu ? Alignment.centerRight : Alignment.centerLeft,
                              child: Text(_strengthText, style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: _strengthColor))),
                          ],

                          const SizedBox(height: 30),
                          PrimaryButton(
                            label: LocalizedStrings.get(context, 'btnSignUp'),
                            icon: Icons.arrow_forward_rounded,
                            onPressed: _demoSignup,
                          ),
                          const SizedBox(height: 20),
                          Center(
                            child: GestureDetector(
                              onTap: () => Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const LoginPage())),
                              child: RichText(text: TextSpan(
                                text: lang == AppLanguage.urdu ? 'پہلے سے اکاؤنٹ ہے؟ ' : (lang == AppLanguage.english ? 'Already have an account? ' : 'Pehle se account hai? '),
                                style: const TextStyle(color: AppColors.textMid, fontSize: 13),
                                children: [TextSpan(
                                  text: lang == AppLanguage.urdu ? 'لاگ ان کریں' : 'Log In',
                                  style: const TextStyle(color: AppColors.primary, fontWeight: FontWeight.w800),
                                )],
                              )),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════
// PAGE 1: WELCOME HOME
// ══════════════════════════════════════════════════════════
class WelcomeHomePage extends StatelessWidget {
  const WelcomeHomePage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: AnimatedBackgroundIcons(
        darkMode: false,
        child: SafeArea(
          child: Column(
            children: [
              Container(
                width: double.infinity,
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
                decoration: const BoxDecoration(
                  gradient: AppGradients.hero,
                  borderRadius: BorderRadius.vertical(bottom: Radius.circular(32)),
                ),
                child: Column(children: [
                  const LanguageSwitcherRow(),
                  const SizedBox(height: 30),
                  Container(
                    width: 80, height: 80,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.18),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(color: Colors.white.withOpacity(0.3), width: 1.5),
                    ),
                    child: const Icon(Icons.handyman_rounded, size: 40, color: Colors.white),
                  ),
                  const SizedBox(height: 14),
                  Text(LocalizedStrings.get(context, 'appName'),
                    style: const TextStyle(fontSize: 24, fontWeight: FontWeight.w900, color: Colors.white)),
                  const SizedBox(height: 6),
                  Text(LocalizedStrings.get(context, 'tagline'),
                    style: TextStyle(fontSize: 13, color: Colors.white.withOpacity(0.75)), textAlign: TextAlign.center),
                ]),
              ),

              const Spacer(),

              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 22),
                child: GestureDetector(
                  onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const LocationPage())),
                  child: Container(
                    padding: const EdgeInsets.symmetric(vertical: 17, horizontal: 20),
                    decoration: BoxDecoration(
                      color: AppColors.surface,
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(color: AppColors.border),
                      boxShadow: [BoxShadow(color: AppColors.cardShadow, blurRadius: 16, offset: const Offset(0, 4))],
                    ),
                    child: Row(children: [
                      Container(
                        padding: const EdgeInsets.all(7),
                        decoration: BoxDecoration(gradient: AppGradients.primary, borderRadius: BorderRadius.circular(10)),
                        child: const Icon(Icons.search_rounded, color: Colors.white, size: 18),
                      ),
                      const SizedBox(width: 14),
                      Expanded(child: Text(LocalizedStrings.get(context, 'searchHint'),
                        style: const TextStyle(fontSize: 13, color: AppColors.textLight, fontWeight: FontWeight.w500))),
                      const Icon(Icons.arrow_forward_ios_rounded, color: AppColors.primary, size: 14),
                    ]),
                  ),
                ),
              ),

              const Spacer(),

              Padding(
                padding: const EdgeInsets.only(bottom: 18),
                child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                  Container(width: 7, height: 7, decoration: const BoxDecoration(color: AppColors.primaryLight, shape: BoxShape.circle)),
                  const SizedBox(width: 8),
                  Text(LocalizedStrings.get(context, 'chatPrompt'),
                    style: const TextStyle(fontSize: 12, color: AppColors.textLight, fontWeight: FontWeight.w500)),
                ]),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════
// PAGE 2: LOCATION SELECTOR
// ══════════════════════════════════════════════════════════
class LocationPage extends StatefulWidget {
  const LocationPage({Key? key}) : super(key: key);
  @override State<LocationPage> createState() => _LocationPageState();
}

class _LocationPageState extends State<LocationPage> {
  String selectedCity = '';
  String selectedArea = '';
  List<String> citiesList = [];
  Map<String, List<String>> cityAreasMap = {};
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _loadCitiesFromBackend();
  }

  Future<void> _loadCitiesFromBackend() async {
    try {
      final citiesData = await BackendClient.instance.fetchCities();
      if (!mounted) return;
      final cities = <String>[];
      final areasMap = <String, List<String>>{};
      for (final c in citiesData) {
        final name = '${c['name'] ?? ''}';
        if (name.isEmpty) continue;
        cities.add(name);
        final rawAreas = c['areas'];
        if (rawAreas is List) {
          areasMap[name] = rawAreas.map((a) => '$a').toList();
        } else {
          areasMap[name] = [];
        }
      }
      setState(() {
        citiesList = cities;
        cityAreasMap = areasMap;
        if (cities.isNotEmpty) {
          selectedCity = cities.first;
          selectedArea = (areasMap[cities.first] ?? []).isNotEmpty
              ? areasMap[cities.first]!.first
              : '';
        }
        _loading = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _loading = false);
    }
  }

  Widget _dropdown({required String value, required List<String> items, required ValueChanged<String?> onChanged, required BuildContext context}) {
    final isUrdu = LanguageConfiguration.of(context)?.currentLanguage == AppLanguage.urdu;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: AppColors.surface, borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
        boxShadow: [BoxShadow(color: AppColors.cardShadow, blurRadius: 8, offset: const Offset(0, 2))],
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: value, isExpanded: true,
          icon: const Icon(Icons.keyboard_arrow_down_rounded, color: AppColors.primary),
          style: const TextStyle(fontSize: 14, color: AppColors.textDark, fontWeight: FontWeight.w600),
          alignment: isUrdu ? Alignment.centerRight : Alignment.centerLeft,
          items: items.map((i) => DropdownMenuItem(value: i,
            child: Align(alignment: isUrdu ? Alignment.centerRight : Alignment.centerLeft,
              child: Text(LocalizedStrings.get(context, i))))).toList(),
          onChanged: onChanged,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isUrdu = LanguageConfiguration.of(context)?.currentLanguage == AppLanguage.urdu;

    return Scaffold(
      backgroundColor: AppColors.bg,
      body: AnimatedBackgroundIcons(
        darkMode: false,
        child: SafeArea(
          child: Column(
            crossAxisAlignment: isUrdu ? CrossAxisAlignment.end : CrossAxisAlignment.start,
            children: [
              Container(
                width: double.infinity,
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
                decoration: const BoxDecoration(
                  gradient: AppGradients.hero,
                  borderRadius: BorderRadius.vertical(bottom: Radius.circular(32)),
                ),
                child: Column(
                  crossAxisAlignment: isUrdu ? CrossAxisAlignment.end : CrossAxisAlignment.start,
                  children: [
                    const LanguageSwitcherRow(),
                    const SizedBox(height: 20),
                    Row(
                      mainAxisAlignment: isUrdu ? MainAxisAlignment.end : MainAxisAlignment.start,
                      children: [
                        if (!isUrdu) IconButton(icon: const Icon(Icons.arrow_back_ios_new_rounded, color: Colors.white, size: 18), onPressed: () => Navigator.pop(context)),
                        Text(LocalizedStrings.get(context, 'locationTitle'),
                          style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w900)),
                        if (isUrdu) IconButton(icon: const Icon(Icons.arrow_back_ios_rounded, color: Colors.white, size: 18), onPressed: () => Navigator.pop(context)),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Padding(
                      padding: EdgeInsets.only(left: isUrdu ? 0 : 8, right: isUrdu ? 8 : 0),
                      child: Text(LocalizedStrings.get(context, 'locationSub'),
                        style: TextStyle(fontSize: 13, color: Colors.white.withOpacity(0.75))),
                    ),
                  ],
                ),
              ),

              Expanded(
                child: _loading
                    ? const Center(child: CircularProgressIndicator(color: AppColors.primary))
                    : citiesList.isEmpty
                        ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
                            const Icon(Icons.cloud_off_rounded, size: 48, color: AppColors.textLight),
                            const SizedBox(height: 12),
                            const Text('Could not load cities.\nCheck your connection and try again.',
                              style: TextStyle(color: AppColors.textLight, fontSize: 13), textAlign: TextAlign.center),
                            const SizedBox(height: 16),
                            ElevatedButton(onPressed: () { setState(() => _loading = true); _loadCitiesFromBackend(); },
                              style: ElevatedButton.styleFrom(backgroundColor: AppColors.primary),
                              child: const Text('Retry', style: TextStyle(color: Colors.white))),
                          ]))
                        : Padding(
                  padding: const EdgeInsets.fromLTRB(20, 28, 20, 20),
                  child: Column(
                    crossAxisAlignment: isUrdu ? CrossAxisAlignment.end : CrossAxisAlignment.start,
                    children: [
                      _label(LocalizedStrings.get(context, 'cityLabel'), Icons.location_city_rounded),
                      const SizedBox(height: 8),
                      _dropdown(value: selectedCity, items: citiesList, context: context,
                        onChanged: (v) => setState(() {
                          selectedCity = v!;
                          final areas = cityAreasMap[selectedCity] ?? [];
                          selectedArea = areas.isNotEmpty ? areas.first : '';
                        })),
                      const SizedBox(height: 24),
                      _label(LocalizedStrings.get(context, 'areaLabel'), Icons.map_outlined),
                      const SizedBox(height: 8),
                      if ((cityAreasMap[selectedCity] ?? []).isNotEmpty)
                        _dropdown(value: selectedArea, items: cityAreasMap[selectedCity]!, context: context,
                          onChanged: (v) => setState(() => selectedArea = v!)),
                      if ((cityAreasMap[selectedCity] ?? []).isEmpty)
                        const Padding(padding: EdgeInsets.symmetric(vertical: 8),
                          child: Text('No areas available', style: TextStyle(color: AppColors.textLight, fontSize: 13))),
                      const Spacer(),
                      PrimaryButton(
                        label: LocalizedStrings.get(context, 'btnNext'),
                        icon: Icons.arrow_forward_rounded,
                        onPressed: selectedCity.isNotEmpty ? () => Navigator.push(context,
                          MaterialPageRoute(builder: (_) => ServicesAndDetailsPage(city: selectedCity, area: selectedArea))) : () {},
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _label(String text, IconData icon) {
    return Row(mainAxisSize: MainAxisSize.min, children: [
      Icon(icon, size: 15, color: AppColors.primary),
      const SizedBox(width: 6),
      Text(text, style: const TextStyle(fontSize: 13, color: AppColors.textMid, fontWeight: FontWeight.w700, letterSpacing: 0.3)),
    ]);
  }
}

// ══════════════════════════════════════════════════════════
// PAGE 3: SERVICES + PROVIDERS
// ══════════════════════════════════════════════════════════
class ServicesAndDetailsPage extends StatefulWidget {
  final String city, area;
  const ServicesAndDetailsPage({Key? key, required this.city, required this.area}) : super(key: key);
  @override State<ServicesAndDetailsPage> createState() => _ServicesAndDetailsPageState();
}

class _ServicesAndDetailsPageState extends State<ServicesAndDetailsPage> {
  String chosenService = '';
  bool   loadingData   = false;
  bool   _loadingCategories = true;
  String? providerError;
  List<ProviderModel> availableList   = [];
  List<ProviderModel> unavailableList = [];
  List<Map<String, dynamic>> _categories = [];

  // Icon mapping — visual config only, not business data.
  // Unknown categories get a generic icon.
  static const Map<String, String> _categoryIcons = {
    'AC Technician': '❄️', 'Appliance Repair': '🔌',
    'Beautician': '💄', 'Carpenter': '🪚',
    'Cleaning Service': '🧹', 'Computer Technician': '💻',
    'Electrician': '⚡', 'Home Tutor': '📖',
    'Mechanic': '🔩', 'Mobile Repair': '📱',
    'Painter': '🎨', 'Plumber': '🔧',
    'Tutor': '📚', 'Water Tank Cleaner': '💧',
    // Legacy name compatibility
    'AC Repair': '❄️',
  };

  static const List<Color> _colorPalette = [
    Color(0xFF0EA5E9), Color(0xFFF59E0B), Color(0xFF22C55E),
    Color(0xFF8B5CF6), Color(0xFFEF4444), Color(0xFF06B6D4),
    Color(0xFFF97316), Color(0xFF10B981), Color(0xFFEC4899),
    Color(0xFF6366F1), Color(0xFF14B8A6), Color(0xFFD946EF),
    Color(0xFF84CC16), Color(0xFF0284C7),
  ];

  @override
  void initState() {
    super.initState();
    _loadCategories();
  }

  Future<void> _loadCategories() async {
    try {
      final cats = await BackendClient.instance.fetchCategories();
      if (!mounted) return;
      setState(() {
        _categories = cats;
        _loadingCategories = false;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() => _loadingCategories = false);
    }
  }

  Future<void> _loadProviders(String type) async {
    setState(() {
      loadingData = true;
      providerError = null;
      chosenService = type;
      availableList.clear();
      unavailableList.clear();
    });

    try {
      final providers = await BackendClient.instance.fetchProviders(
        query: type,
        city: widget.city,
        area: widget.area,
      );
      if (!mounted) return;
      setState(() {
        loadingData = false;
        availableList = providers.where((provider) => provider.isAvailable).toList();
        unavailableList = providers.where((provider) => !provider.isAvailable).toList();
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        loadingData = false;
        providerError = 'Backend unavailable. Check PROFIXER_API_BASE_URL and server health.';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final isUrdu = LanguageConfiguration.of(context)?.currentLanguage == AppLanguage.urdu;
    final loc    = "${LocalizedStrings.get(context, widget.area)}, ${LocalizedStrings.get(context, widget.city)}";

    return Scaffold(
      backgroundColor: AppColors.bg,
      body: AnimatedBackgroundIcons(
        darkMode: false,
        child: SafeArea(
          child: Column(
            crossAxisAlignment: isUrdu ? CrossAxisAlignment.end : CrossAxisAlignment.start,
            children: [
              Container(
                width: double.infinity,
                padding: const EdgeInsets.fromLTRB(20, 16, 20, 24),
                decoration: const BoxDecoration(
                  gradient: AppGradients.hero,
                  borderRadius: BorderRadius.vertical(bottom: Radius.circular(32)),
                ),
                child: Column(
                  crossAxisAlignment: isUrdu ? CrossAxisAlignment.end : CrossAxisAlignment.start,
                  children: [
                    const LanguageSwitcherRow(),
                    const SizedBox(height: 18),
                    Row(
                      mainAxisAlignment: isUrdu ? MainAxisAlignment.end : MainAxisAlignment.start,
                      children: [
                        if (!isUrdu) IconButton(icon: const Icon(Icons.arrow_back_ios_new_rounded, color: Colors.white, size: 18), onPressed: () => Navigator.pop(context)),
                        const Icon(Icons.location_on_rounded, color: AppColors.accent, size: 18),
                        const SizedBox(width: 6),
                        Text(loc, style: const TextStyle(color: Colors.white, fontSize: 15, fontWeight: FontWeight.w800)),
                        if (isUrdu) IconButton(icon: const Icon(Icons.arrow_back_ios_rounded, color: Colors.white, size: 18), onPressed: () => Navigator.pop(context)),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Text(LocalizedStrings.get(context, 'selectServiceTitle'),
                      style: TextStyle(color: Colors.white.withOpacity(0.85), fontSize: 13, fontWeight: FontWeight.w600)),
                    const SizedBox(height: 12),
                    if (_loadingCategories)
                      const SizedBox(height: 72, child: Center(child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2)))
                    else if (_categories.isEmpty)
                      const SizedBox(height: 72, child: Center(child: Text('No services available', style: TextStyle(color: Colors.white70, fontSize: 12))))
                    else
                      SizedBox(
                        height: 80,
                        child: ListView.separated(
                          scrollDirection: Axis.horizontal,
                          itemCount: _categories.length,
                          separatorBuilder: (_, __) => const SizedBox(width: 8),
                          itemBuilder: (context, i) {
                            final catName = '${_categories[i]['name'] ?? ''}';
                            final emoji = _categoryIcons[catName] ?? '🛠️';
                            final color = _colorPalette[i % _colorPalette.length];
                            return SizedBox(width: 90, child: _serviceBtn(catName, emoji, color));
                          },
                        ),
                      ),
                  ],
                ),
              ),

              Expanded(
                child: chosenService.isEmpty
                    ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
                        Icon(Icons.touch_app_rounded, size: 52, color: AppColors.primary.withOpacity(0.3)),
                        const SizedBox(height: 12),
                        Text(LocalizedStrings.get(context, 'selectServicePrompt'),
                          style: const TextStyle(color: AppColors.textLight, fontSize: 13), textAlign: TextAlign.center),
                      ]))
                    : loadingData
                        ? const Center(child: CircularProgressIndicator(color: AppColors.primary))
                        : providerError != null
                            ? Center(
                                child: Padding(
                                  padding: const EdgeInsets.all(24),
                                  child: Text(
                                    providerError!,
                                    style: const TextStyle(color: AppColors.textMid, fontSize: 13),
                                    textAlign: TextAlign.center,
                                  ),
                                ),
                              )
                        : ListView(
                            padding: const EdgeInsets.fromLTRB(16, 20, 16, 20),
                            children: [
                              _sectionHeader(LocalizedStrings.get(context, 'availableNowTitle'), AppColors.primary, Icons.check_circle_rounded, isUrdu),
                              if (availableList.isNotEmpty) _availableCard(availableList.first),
                              if (availableList.isEmpty && unavailableList.isEmpty)
                                Padding(
                                  padding: const EdgeInsets.symmetric(vertical: 24),
                                  child: Text(
                                    'No providers found for $chosenService in ${widget.area.isNotEmpty ? '${widget.area}, ' : ''}${widget.city}.\n'
                                    'Try selecting a different area or search across the whole city.',
                                    style: const TextStyle(color: AppColors.textLight, fontSize: 13),
                                    textAlign: TextAlign.center,
                                  ),
                                ),
                              const SizedBox(height: 22),
                              _sectionHeader(LocalizedStrings.get(context, 'busyTitle'), AppColors.textMid, Icons.hourglass_top_rounded, isUrdu),
                              ...unavailableList.map(_busyCard),
                            ],
                          ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionHeader(String text, Color color, IconData icon, bool isUrdu) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        mainAxisAlignment: isUrdu ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: isUrdu
            ? [Text(text, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w800, color: color)), const SizedBox(width: 8), Icon(icon, size: 16, color: color)]
            : [Icon(icon, size: 16, color: color), const SizedBox(width: 8), Text(text, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w800, color: color))],
      ),
    );
  }

  Widget _availableCard(ProviderModel p) {
    final actionLabel = p.phone.isEmpty ? 'View Details' : 'Call ${p.phone}';
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface, borderRadius: BorderRadius.circular(20),
        border: Border.all(color: const Color(0xFF22C55E).withOpacity(0.3)),
        boxShadow: [BoxShadow(color: AppColors.cardShadow, blurRadius: 14, offset: const Offset(0, 4))],
      ),
      child: Column(children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 12),
          child: Row(children: [
            Container(
              width: 50, height: 50,
              decoration: BoxDecoration(
                gradient: AppGradients.primary, shape: BoxShape.circle,
                boxShadow: [BoxShadow(color: AppColors.primary.withOpacity(0.3), blurRadius: 10, offset: const Offset(0, 3))],
              ),
              child: Center(child: Text(p.name[0], style: const TextStyle(color: Colors.white, fontSize: 20, fontWeight: FontWeight.w900))),
            ),
            const SizedBox(width: 14),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(p.name, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 15, color: AppColors.textDark)),
              const SizedBox(height: 3),
              Text(p.specialty, style: const TextStyle(fontSize: 12, color: AppColors.textMid)),
              const SizedBox(height: 6),
              Row(children: [
                const Icon(Icons.star_rounded, color: AppColors.accent, size: 14),
                const SizedBox(width: 3),
                Text('${p.rating}', style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: AppColors.textDark)),
                const SizedBox(width: 6),
                Text('(${p.reviews} reviews)', style: const TextStyle(fontSize: 11, color: AppColors.textLight)),
              ]),
            ])),
            Column(children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: const Color(0xFF22C55E).withOpacity(0.12), borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: const Color(0xFF22C55E).withOpacity(0.4)),
                ),
                child: const Text('🟢 Active', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: Color(0xFF16A34A))),
              ),
              const SizedBox(height: 6),
              Row(mainAxisSize: MainAxisSize.min, children: [
                const Icon(Icons.access_time_rounded, size: 12, color: AppColors.textLight),
                const SizedBox(width: 3),
                Text('${p.eta} min', style: const TextStyle(fontSize: 11, color: AppColors.textMid, fontWeight: FontWeight.w600)),
              ]),
            ]),
          ]),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: Container(
            width: double.infinity, height: 44,
            decoration: BoxDecoration(
              gradient: const LinearGradient(colors: [Color(0xFF16A34A), Color(0xFF22C55E)]),
              borderRadius: BorderRadius.circular(12),
              boxShadow: [BoxShadow(color: const Color(0xFF22C55E).withOpacity(0.35), blurRadius: 10, offset: const Offset(0, 4))],
            ),
            child: Material(
              color: Colors.transparent,
              child: InkWell(
                borderRadius: BorderRadius.circular(12),
                onTap: () => _showProviderDetailDialog(p),
                child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                  Icon(p.phone.isEmpty ? Icons.info_rounded : Icons.phone_rounded, color: Colors.white, size: 18),
                  const SizedBox(width: 8),
                  Text(actionLabel, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 13)),
                ]),
              ),
            ),
          ),
        ),
      ]),
    );
  }

  void _showProviderDetailDialog(ProviderModel p) {
    showDialog(
      context: context,
      builder: (ctx) => Dialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        child: Container(
          padding: const EdgeInsets.all(24),
          constraints: const BoxConstraints(maxWidth: 360),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            // Avatar
            Container(
              width: 64, height: 64,
              decoration: BoxDecoration(gradient: AppGradients.primary, shape: BoxShape.circle),
              child: Center(child: Text(p.name[0], style: const TextStyle(color: Colors.white, fontSize: 28, fontWeight: FontWeight.w900))),
            ),
            const SizedBox(height: 14),
            Text(p.name, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: AppColors.textDark), textAlign: TextAlign.center),
            const SizedBox(height: 4),
            Text(p.specialty, style: const TextStyle(fontSize: 13, color: AppColors.textMid)),
            const SizedBox(height: 16),
            // Info grid
            _detailRow(Icons.star_rounded, 'Rating', '${p.rating} ⭐'),
            _detailRow(Icons.work_rounded, 'Jobs Done', '${p.reviews}'),
            _detailRow(Icons.access_time_rounded, 'Response', '${p.eta} min'),
            _detailRow(Icons.location_city_rounded, 'Location', '${p.area}, ${p.city}'),
            _detailRow(Icons.attach_money_rounded, 'Price Range', p.priceRange.isEmpty ? 'N/A' : p.priceRange),
            _detailRow(Icons.circle, 'Availability', p.availability.isEmpty ? 'N/A' : p.availability),
            if (p.phone.isNotEmpty) _detailRow(Icons.phone_rounded, 'Phone', p.phone),
            const SizedBox(height: 20),
            // Action buttons
            Row(children: [
              Expanded(
                child: Container(
                  height: 44,
                  decoration: BoxDecoration(
                    gradient: AppGradients.primary,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Material(
                    color: Colors.transparent,
                    child: InkWell(
                      borderRadius: BorderRadius.circular(12),
                      onTap: () {
                        Navigator.pop(ctx);
                        // Open chatbot to initiate booking via conversation
                        showModalBottomSheet(
                          context: context, isScrollControlled: true, backgroundColor: Colors.transparent,
                          builder: (_) => const ChatBotWidget(),
                        );
                      },
                      child: const Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                        Icon(Icons.chat_rounded, color: Colors.white, size: 16),
                        SizedBox(width: 6),
                        Text('Book via Chat', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 12)),
                      ]),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Container(
                height: 44, width: 44,
                decoration: BoxDecoration(
                  color: AppColors.bg,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppColors.primary.withOpacity(0.2)),
                ),
                child: Material(
                  color: Colors.transparent,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(12),
                    onTap: () {
                      // Open in Google Maps — no API key required
                      final query = '${p.area}, ${p.city}';
                      final url = Uri.parse('https://www.google.com/maps/search/?api=1&query=${Uri.encodeComponent(query)}');
                      launchUrl(url, mode: LaunchMode.externalApplication).catchError((_) => false);
                    },
                    child: const Icon(Icons.map_rounded, color: AppColors.primary, size: 20),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Container(
                height: 44, width: 44,
                decoration: BoxDecoration(
                  color: AppColors.bg,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: AppColors.border),
                ),
                child: Material(
                  color: Colors.transparent,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(12),
                    onTap: () => Navigator.pop(ctx),
                    child: const Icon(Icons.close_rounded, color: AppColors.textMid, size: 20),
                  ),
                ),
              ),
            ]),
          ]),
        ),
      ),
    );
  }

  Widget _detailRow(IconData icon, String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(children: [
        Icon(icon, size: 16, color: AppColors.primary),
        const SizedBox(width: 10),
        SizedBox(
          width: 80,
          child: Text(label, style: const TextStyle(fontSize: 12, color: AppColors.textLight, fontWeight: FontWeight.w600)),
        ),
        Expanded(child: Text(value, style: const TextStyle(fontSize: 12, color: AppColors.textDark, fontWeight: FontWeight.w700))),
      ]),
    );
  }

  Widget _busyCard(ProviderModel p) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface, borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
        boxShadow: [BoxShadow(color: AppColors.cardShadow, blurRadius: 8, offset: const Offset(0, 2))],
      ),
      child: Row(children: [
        Container(
          width: 44, height: 44,
          decoration: BoxDecoration(color: AppColors.textLight.withOpacity(0.1), shape: BoxShape.circle),
          child: Center(child: Text(p.name[0], style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: AppColors.textMid))),
        ),
        const SizedBox(width: 12),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(p.name, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 14, color: AppColors.textDark)),
          const SizedBox(height: 3),
          Text(p.specialty, style: const TextStyle(fontSize: 12, color: AppColors.textLight)),
        ])),
        Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(color: AppColors.accent.withOpacity(0.12), borderRadius: BorderRadius.circular(10)),
            child: const Text('Busy', style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: Color(0xFFD97706))),
          ),
          const SizedBox(height: 4),
          Text('in ${p.nextAvailableIn}', style: const TextStyle(fontSize: 11, color: AppColors.textMid, fontWeight: FontWeight.w600)),
        ]),
      ]),
    );
  }

  Widget _serviceBtn(String name, String emoji, Color activeColor) {
    final active = chosenService == name;
    return GestureDetector(
      onTap: () => _loadProviders(name),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        height: 72,
        decoration: BoxDecoration(
          color: active ? activeColor.withOpacity(0.15) : Colors.white.withOpacity(0.10),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: active ? activeColor : Colors.white.withOpacity(0.2), width: active ? 2 : 1),
        ),
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          Text(emoji, style: const TextStyle(fontSize: 22)),
          const SizedBox(height: 4),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: Text(LocalizedStrings.get(context, name),
              textAlign: TextAlign.center, maxLines: 2, overflow: TextOverflow.ellipsis,
              style: TextStyle(fontSize: 9, fontWeight: FontWeight.w700,
                color: active ? activeColor : Colors.white.withOpacity(0.8))),
          ),
        ]),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════
// CHATBOT WIDGET
// ══════════════════════════════════════════════════════════
class ChatBotWidget extends StatefulWidget {
  const ChatBotWidget({Key? key}) : super(key: key);
  @override State<ChatBotWidget> createState() => _ChatBotWidgetState();
}

class _ChatBotWidgetState extends State<ChatBotWidget> {
  final List<Map<String, dynamic>> _messages = [];
  final TextEditingController _ctrl = TextEditingController();
  bool _isSending = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (_messages.isEmpty) {
      final lang = LanguageConfiguration.of(context)?.currentLanguage ?? AppLanguage.romanUrdu;
      String g = "Assalam-o-Alaikum! Main Assistant hoon. 🤖 Main Roman Urdu, Urdu aur English teeno samajh sakta hoon. Bataiye kya kaam hai?";
      if (lang == AppLanguage.english) g = "Hello! I am your AI Agent. 🤖 I can support English, Urdu, and Roman Urdu. How can I help you?";
      if (lang == AppLanguage.urdu)    g = "السلام علیکم! میں آپ کا اے آئی اسسٹنٹ ہوں۔ 🤖 میں اردو، انگلش اور رومن اردو سمجھ سکتا ہوں۔ بتائیے کیا خدمت کروں؟";
      _messages.add({"sender": "bot", "text": g});
    }
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    if (_ctrl.text.trim().isEmpty || _isSending) return;
    final txt = _ctrl.text.trim();
    setState(() {
      _messages.add({"sender": "user", "text": txt});
      _isSending = true;
      _ctrl.clear();
    });

    try {
      final result = await BackendClient.instance.sendChat(txt);
      if (!mounted) return;
      setState(() {
        // Add AI response text
        _messages.add({
          "sender": "bot",
          "text": result.response,
          "handoffTrace": result.handoffTrace,
        });

        // If booking info is present, render a styled booking card
        if (result.booking != null) {
          final b = result.booking!;
          final statusEmoji = result.isBookingConfirmed ? '✅' : (result.isBookingCancelled ? '❌' : '📋');
          final bookingInfo = StringBuffer();
          bookingInfo.writeln('$statusEmoji Booking ${result.status.replaceAll("_", " ").toUpperCase()}');
          bookingInfo.writeln('───────────────');
          if (b['service_type'] != null) bookingInfo.writeln('🛠️ Service: ${b['service_type']}');
          if (b['provider_name'] != null) bookingInfo.writeln('👤 Provider: ${b['provider_name']}');
          if (b['location'] != null) bookingInfo.writeln('📍 Location: ${b['location']}');
          if (b['scheduled_time'] != null) bookingInfo.writeln('🕒 Time: ${b['scheduled_time']}');
          if (b['status'] != null) bookingInfo.writeln('📊 Status: ${b['status']}');
          if (b['booking_id'] != null) bookingInfo.writeln('🆔 ID: ${b['booking_id']}');
          _messages.add({"sender": "booking", "text": bookingInfo.toString().trimRight()});
        }
      });
    } catch (error) {
      if (!mounted) return;
      final errMsg = error is StateError ? error.message : 'Connection failed. Please try again.';
      setState(() => _messages.add({"sender": "bot", "text": errMsg}));
    } finally {
      if (mounted) setState(() => _isSending = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      child: Container(
        height: MediaQuery.of(context).size.height * 0.62,
        decoration: const BoxDecoration(color: AppColors.surface, borderRadius: BorderRadius.vertical(top: Radius.circular(28))),
        child: Column(children: [
          const SizedBox(height: 10),
          Container(width: 42, height: 4, decoration: BoxDecoration(color: AppColors.border, borderRadius: BorderRadius.circular(4))),
          Container(
            margin: const EdgeInsets.only(top: 12),
            padding: const EdgeInsets.fromLTRB(20, 14, 16, 14),
            decoration: const BoxDecoration(gradient: AppGradients.primary),
            child: Row(children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), borderRadius: BorderRadius.circular(10)),
                child: const Icon(Icons.smart_toy_rounded, color: Colors.white, size: 20),
              ),
              const SizedBox(width: 12),
              const Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('AI Multi-Lang Orchestrator', style: TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 14)),
                Text('Online • Tri-lingual support', style: TextStyle(color: Colors.white60, fontSize: 11)),
              ])),
              IconButton(icon: const Icon(Icons.close_rounded, color: Colors.white), onPressed: () => Navigator.pop(context)),
            ]),
          ),

          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, i) {
                final msg = _messages[i];
                final isUser = msg["sender"] == "user";
                final isBooking = msg["sender"] == "booking";

                if (isBooking) {
                  // Styled booking info card
                  return Align(
                    alignment: Alignment.centerLeft,
                    child: Container(
                      margin: const EdgeInsets.only(bottom: 10),
                      padding: const EdgeInsets.all(14),
                      constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.80),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFFF0FDF4), Color(0xFFECFDF5)],
                          begin: Alignment.topLeft, end: Alignment.bottomRight,
                        ),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: const Color(0xFF22C55E).withOpacity(0.3)),
                        boxShadow: [BoxShadow(color: AppColors.cardShadow, blurRadius: 8, offset: const Offset(0, 2))],
                      ),
                      child: Text(msg["text"]!, style: const TextStyle(color: AppColors.textDark, fontSize: 12, height: 1.5, fontWeight: FontWeight.w600)),
                    ),
                  );
                }

                final hasTrace = !isUser && !isBooking && msg["handoffTrace"] != null && (msg["handoffTrace"] as List).isNotEmpty;

                return Align(
                  alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                  child: Column(
                    crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
                    children: [
                      Container(
                        margin: const EdgeInsets.only(bottom: 10),
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                        constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
                        decoration: BoxDecoration(
                          gradient: isUser ? AppGradients.primary : null,
                          color: isUser ? null : AppColors.bg,
                          borderRadius: BorderRadius.only(
                            topLeft: const Radius.circular(16), topRight: const Radius.circular(16),
                            bottomLeft: Radius.circular(isUser ? 16 : 4),
                            bottomRight: Radius.circular(isUser ? 4 : 16),
                          ),
                          boxShadow: [BoxShadow(color: AppColors.cardShadow, blurRadius: 6, offset: const Offset(0, 2))],
                        ),
                        child: Text(msg["text"]!, style: TextStyle(color: isUser ? Colors.white : AppColors.textDark, fontSize: 13, height: 1.4)),
                      ),
                      if (hasTrace)
                        Container(
                          margin: const EdgeInsets.only(left: 4, bottom: 10),
                          constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
                          decoration: BoxDecoration(
                            color: Colors.grey.shade50,
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: Colors.grey.shade200),
                          ),
                          child: Theme(
                            data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
                            child: ExpansionTile(
                              dense: true,
                              title: Row(
                                children: [
                                  Icon(Icons.alt_route_rounded, size: 14, color: AppColors.primary),
                                  SizedBox(width: 6),
                                  Text(
                                    'Agent Workflow',
                                    style: TextStyle(
                                      fontSize: 11,
                                      fontWeight: FontWeight.bold,
                                      color: AppColors.textDark,
                                    ),
                                  ),
                                ],
                              ),
                              children: (msg["handoffTrace"] as List).map<Widget>((entry) {
                                final map = entry as Map<String, dynamic>;
                                final agent = map["to_agent"] ?? "Agent";
                                final task = map["task"] ?? "";
                                final status = map["status"] ?? "";
                                final summary = map["summary"] ?? "";

                                final Color statusColor = status == 'blocked'
                                    ? Colors.red
                                    : (status == 'passed' || status == 'completed'
                                        ? Colors.green
                                        : Colors.orange);
                                
                                final IconData statusIcon = status == 'blocked'
                                    ? Icons.cancel_rounded
                                    : (status == 'passed' || status == 'completed'
                                        ? Icons.check_circle_rounded
                                        : Icons.hourglass_empty_rounded);

                                return Padding(
                                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                                  child: Column(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      Row(
                                        children: [
                                          Icon(statusIcon, size: 12, color: statusColor),
                                          const SizedBox(width: 6),
                                          Text(
                                            agent.replaceAll("_", " ").toUpperCase(),
                                            style: const TextStyle(
                                              fontSize: 10,
                                              fontWeight: FontWeight.w700,
                                              color: AppColors.textDark,
                                            ),
                                          ),
                                          const Spacer(),
                                          Text(
                                            status.toUpperCase(),
                                            style: TextStyle(
                                              fontSize: 9,
                                              fontWeight: FontWeight.w800,
                                              color: statusColor,
                                            ),
                                          ),
                                        ],
                                      ),
                                      const SizedBox(height: 2),
                                      Padding(
                                        padding: const EdgeInsets.only(left: 18),
                                        child: Text(
                                          '$task\n$summary'.trim(),
                                          style: TextStyle(
                                            fontSize: 9.5,
                                            color: Colors.grey.shade600,
                                            height: 1.3,
                                          ),
                                        ),
                                      ),
                                      const Divider(height: 8),
                                    ],
                                  ),
                                );
                              }).toList(),
                            ),
                          ),
                        ),
                    ],
                  ),
                );
              },
            ),
          ),

          Container(
            padding: const EdgeInsets.fromLTRB(16, 10, 12, 14),
            decoration: BoxDecoration(color: AppColors.surface, border: Border(top: BorderSide(color: AppColors.border))),
            child: Row(children: [
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  decoration: BoxDecoration(color: AppColors.bg, borderRadius: BorderRadius.circular(24), border: Border.all(color: AppColors.border)),
                  child: TextField(
                    controller: _ctrl,
                    decoration: const InputDecoration(
                      hintText: 'AC kharab hai / I need plumber',
                      hintStyle: TextStyle(color: AppColors.textLight, fontSize: 12),
                      border: InputBorder.none, contentPadding: EdgeInsets.symmetric(vertical: 11),
                    ),
                    onSubmitted: (_) => _send(),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Container(
                width: 44, height: 44,
                decoration: BoxDecoration(
                  gradient: AppGradients.primary, shape: BoxShape.circle,
                  boxShadow: [BoxShadow(color: AppColors.primary.withOpacity(0.4), blurRadius: 10, offset: const Offset(0, 3))],
                ),
                child: Material(
                  color: Colors.transparent,
                  child: InkWell(
                    borderRadius: BorderRadius.circular(22), onTap: _send,
                    child: _isSending
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                          )
                        : const Icon(Icons.send_rounded, color: Colors.white, size: 20),
                  ),
                ),
              ),
            ]),
          ),
        ]),
      ),
    );
  }
}
