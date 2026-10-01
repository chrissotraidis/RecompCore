// SPDX-License-Identifier: GPL-3.0-or-later
#include "gxruntime/memory_card.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

#define DOL_CARD_CONTAINER_VERSION 1u
#define DOL_CARD_HEADER_SIZE 40u
#define DOL_CARD_RECORD_SIZE 68u

static const u8 k_card_magic[8] = {'D', 'O', 'L', 'C', 'A', 'R', 'D', '1'};

typedef struct DolMemoryCardFile {
    bool used;
    DolMemoryCardStat stat;
    u8* data;
} DolMemoryCardFile;

struct DolMemoryCard {
    char* path;
#ifdef _WIN32
    HANDLE lock_handle;
#else
    int lock_fd;
#endif
    u16 size_mbits;
    u16 encoding;
    u64 serial;
    u8 game_code[4];
    u8 company[2];
    bool mounted;
    s32 last_result;
    u32 transferred_bytes;
    DolMemoryCardFile files[DOL_CARD_MAX_FILES];
};

static bool valid_size_mbits(u16 size_mbits) {
    switch (size_mbits) {
    case 4:
    case 8:
    case 16:
    case 32:
    case 64:
    case 128:
        return true;
    default:
        return false;
    }
}

static u32 card_total_blocks(const DolMemoryCard* card) {
    return (u32)card->size_mbits * 16u;
}

static u32 card_user_blocks(const DolMemoryCard* card) {
    return card_total_blocks(card) - DOL_CARD_SYSTEM_BLOCKS;
}

static u32 file_blocks(u32 length) {
    return (length + DOL_CARD_SECTOR_SIZE - 1u) / DOL_CARD_SECTOR_SIZE;
}

static u32 hash_bytes(const u8* data, size_t size) {
    u32 hash = 2166136261u;
    for (size_t i = 0; i < size; i++)
        hash = (hash ^ data[i]) * 16777619u;
    return hash;
}

static u64 generated_serial(const char* path) {
    u64 hash = 1469598103934665603ull;
    if (path != NULL) {
        for (const u8* p = (const u8*)path; *p != 0; p++)
            hash = (hash ^ *p) * 1099511628211ull;
    }
    hash ^= (u64)time(NULL) * 0x9E3779B97F4A7C15ull;
    return hash != 0 ? hash : 0x444F4C4341524431ull;
}

static char* duplicate_string(const char* source) {
    if (source == NULL || source[0] == '\0')
        return NULL;
    size_t length = strlen(source) + 1u;
    char* copy = (char*)malloc(length);
    if (copy != NULL)
        memcpy(copy, source, length);
    return copy;
}

static void clear_files(DolMemoryCard* card) {
    for (u32 i = 0; i < DOL_CARD_MAX_FILES; i++) {
        free(card->files[i].data);
        memset(&card->files[i], 0, sizeof(card->files[i]));
    }
}

static DolCardResult set_result(DolMemoryCard* card, DolCardResult result,
                                u32 transferred_bytes) {
    if (card != NULL) {
        card->last_result = result;
        card->transferred_bytes = transferred_bytes;
    }
    return result;
}

static bool card_is_mounted(DolMemoryCard* card) {
    return card != NULL && card->mounted;
}

static bool file_matches_game(const DolMemoryCard* card,
                              const DolMemoryCardFile* file) {
    return memcmp(file->stat.game_code, card->game_code,
                  sizeof card->game_code) == 0 &&
           memcmp(file->stat.company, card->company,
                  sizeof card->company) == 0;
}

static s32 find_file(const DolMemoryCard* card, const char* name) {
    if (name == NULL)
        return -1;
    for (u32 i = 0; i < DOL_CARD_MAX_FILES; i++) {
        const DolMemoryCardFile* file = &card->files[i];
        if (file->used && file_matches_game(card, file) &&
            strcmp(file->stat.file_name, name) == 0)
            return (s32)i;
    }
    return -1;
}

static u32 payload_size(const DolMemoryCard* card, u32* file_count) {
    u32 count = 0;
    u32 size = 0;
    for (u32 i = 0; i < DOL_CARD_MAX_FILES; i++) {
        if (!card->files[i].used)
            continue;
        count++;
        size += DOL_CARD_RECORD_SIZE + card->files[i].stat.length;
    }
    if (file_count != NULL)
        *file_count = count;
    return size;
}

static bool flush_card_file(FILE* file) {
    if (fflush(file) != 0)
        return false;
#ifdef _WIN32
    return _commit(_fileno(file)) == 0;
#elif defined(__APPLE__)
    return fcntl(fileno(file), F_FULLFSYNC) == 0;
#else
    return fsync(fileno(file)) == 0;
#endif
}

#ifndef _WIN32
static int open_card_directory(const char* path) {
    char* parent = duplicate_string(path);
    if (parent == NULL)
        return -1;
    char* slash = strrchr(parent, '/');
    if (slash == NULL)
        strcpy(parent, ".");
    else if (slash == parent)
        slash[1] = '\0';
    else
        *slash = '\0';
    int fd = open(parent, O_RDONLY | O_DIRECTORY);
    free(parent);
    return fd;
}
#endif

static bool lock_card(DolMemoryCard* card) {
    if (card->path == NULL)
        return true;
    size_t size = strlen(card->path) + 6u;
    char* path = (char*)malloc(size);
    if (path == NULL)
        return false;
    snprintf(path, size, "%s.lock", card->path);
#ifdef _WIN32
    HANDLE handle = CreateFileA(path, GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    OVERLAPPED offset = {0};
    bool ok = handle != INVALID_HANDLE_VALUE && LockFileEx(handle,
        LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &offset);
    if (ok)
        card->lock_handle = handle;
    else if (handle != INVALID_HANDLE_VALUE)
        CloseHandle(handle);
#else
    int fd = open(path, O_CREAT | O_RDWR, 0600);
    bool ok = fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0;
    if (ok)
        card->lock_fd = fd;
    else if (fd >= 0)
        close(fd);
#endif
    free(path);
    if (!ok)
        fprintf(stderr, "[card] cannot lock %s: another BlueWake session may be using this card\n", card->path);
    return ok;
}

static bool backup_card(const DolMemoryCard* card) {
    FILE* source = fopen(card->path, "rb");
    if (source == NULL)
        return errno == ENOENT;
    size_t capacity = strlen(card->path) + 64u;
    char* temporary = (char*)malloc(capacity);
    char* backup = (char*)malloc(capacity);
    if (temporary == NULL || backup == NULL) {
        fclose(source); free(temporary); free(backup); return false;
    }
    snprintf(temporary, capacity, "%s.backup-tmp", card->path);
    snprintf(backup, capacity, "%s.bak", card->path);
    FILE* output = fopen(temporary, "wb");
    bool ok = output != NULL;
    u8 buffer[8192];
    size_t count;
    while (ok && (count = fread(buffer, 1, sizeof buffer, source)) != 0)
        ok = fwrite(buffer, 1, count, output) == count;
    if (ferror(source)) ok = false;
    if (fclose(source) != 0) ok = false;
    if (output != NULL) {
        if (ok && !flush_card_file(output)) ok = false;
        if (fclose(output) != 0) ok = false;
    }
    if (ok && rename(temporary, backup) != 0) ok = false;
    if (!ok) remove(temporary);
    // Seven fixed daily slots bound storage; update a slot only once per UTC
    // day. Rotation uses atomic replacement, just like the live card.
    time_t now = time(NULL);
    unsigned day = (unsigned)(now / 86400);
    snprintf(temporary, capacity, "%s.bak-day-%u", card->path, day % 7u);
    struct stat status;
    if (ok && (stat(temporary, &status) != 0 || (unsigned)(status.st_mtime / 86400) != day)) {
        FILE* daily = fopen(temporary, "rb");
        if (daily != NULL) fclose(daily);
        // Copy the already flushed .bak, then atomically rotate the daily slot.
        source = fopen(backup, "rb");
        snprintf(backup, capacity, "%s.daily-tmp", card->path);
        output = source != NULL ? fopen(backup, "wb") : NULL;
        ok = source != NULL && output != NULL;
        while (ok && (count = fread(buffer, 1, sizeof buffer, source)) != 0)
            ok = fwrite(buffer, 1, count, output) == count;
        if (source != NULL) { if (ferror(source)) ok = false; if (fclose(source) != 0) ok = false; }
        if (output != NULL) { if (ok && !flush_card_file(output)) ok = false; if (fclose(output) != 0) ok = false; }
        if (ok && rename(backup, temporary) != 0) ok = false;
        if (!ok) remove(backup);
    }
    free(temporary); free(backup);
    return ok;
}

typedef enum {
    CARD_WRITE_FAILED,
    CARD_WRITE_PUBLISHED,
    CARD_WRITE_SYNCED
} CardWriteResult;

static CardWriteResult write_container(const DolMemoryCard* card) {
    if (card->path == NULL)
        return CARD_WRITE_SYNCED;

    u32 file_count = 0;
    u32 body_size = payload_size(card, &file_count);
    size_t total_size = (size_t)DOL_CARD_HEADER_SIZE + body_size;
    u8* bytes = (u8*)calloc(1, total_size);
    if (bytes == NULL)
        return CARD_WRITE_FAILED;

    memcpy(bytes, k_card_magic, sizeof k_card_magic);
    write_be32(bytes + 8, DOL_CARD_CONTAINER_VERSION);
    write_be16(bytes + 12, card->size_mbits);
    write_be16(bytes + 14, card->encoding);
    write_be32(bytes + 16, DOL_CARD_SECTOR_SIZE);
    write_be64(bytes + 20, card->serial);
    write_be32(bytes + 28, file_count);
    write_be32(bytes + 32, body_size);

    u8* record = bytes + DOL_CARD_HEADER_SIZE;
    for (u32 i = 0; i < DOL_CARD_MAX_FILES; i++) {
        const DolMemoryCardFile* file = &card->files[i];
        if (!file->used)
            continue;

        write_be16(record, (u16)i);
        write_be32(record + 4, file->stat.length);
        write_be32(record + 8, file->stat.time);
        memcpy(record + 12, file->stat.file_name, DOL_CARD_FILENAME_MAX);
        memcpy(record + 44, file->stat.game_code, 4);
        memcpy(record + 48, file->stat.company, 2);
        record[50] = file->stat.banner_format;
        record[51] = file->stat.permission;
        write_be32(record + 52, file->stat.icon_address);
        write_be16(record + 56, file->stat.icon_format);
        write_be16(record + 58, file->stat.icon_speed);
        write_be32(record + 60, file->stat.comment_address);
        write_be32(record + 64,
                   hash_bytes(file->data, file->stat.length));
        memcpy(record + DOL_CARD_RECORD_SIZE, file->data, file->stat.length);
        record += DOL_CARD_RECORD_SIZE + file->stat.length;
    }
    write_be32(bytes + 36,
               hash_bytes(bytes + DOL_CARD_HEADER_SIZE, body_size));

    size_t path_length = strlen(card->path);
    char* temporary_path = (char*)malloc(path_length + 40u);
    if (temporary_path == NULL) {
        free(bytes);
        return CARD_WRITE_FAILED;
    }
    snprintf(temporary_path, path_length + 40u, "%s.tmp-%lu", card->path,
#ifdef _WIN32
             (unsigned long)GetCurrentProcessId()
#else
             (unsigned long)getpid()
#endif
    );

    bool success = false;
    FILE* output = fopen(temporary_path, "wb");
    if (output != NULL) {
        success = fwrite(bytes, 1, total_size, output) == total_size;
        if (success && !flush_card_file(output))
            success = false;
        if (fclose(output) != 0)
            success = false;
        output = NULL;
    }

#ifndef _WIN32
    int directory_fd = success ? open_card_directory(card->path) : -1;
    if (success && directory_fd < 0)
        success = false;
#endif
    if (success && !backup_card(card))
        success = false;
    // POSIX rename and the Windows replacement shim preserve the old card
    // on failure. Never delete the destination to retry a failed replacement.
    bool published = false;
    if (success) {
        published = rename(temporary_path, card->path) == 0;
        success = published;
    }
#ifndef _WIN32
    if (success && fsync(directory_fd) != 0)
        success = false;
    if (directory_fd >= 0 && close(directory_fd) != 0)
        success = false;
#endif
    if (!success)
        remove(temporary_path);

    free(temporary_path);
    free(bytes);
    // After publication, an uncertain directory flush is still an I/O error,
    // but callers must retain the published contents in memory.
    return success ? CARD_WRITE_SYNCED : published ? CARD_WRITE_PUBLISHED : CARD_WRITE_FAILED;
}

static bool read_whole_file(const char* path, u8** data, size_t* size) {
    FILE* input = fopen(path, "rb");
    if (input == NULL)
        return false;
    if (fseek(input, 0, SEEK_END) != 0) {
        fclose(input);
        return false;
    }
    long end = ftell(input);
    if (end < 0 || end > (128 * 16 * 8192 + 127 * 68 + 40) ||
        fseek(input, 0, SEEK_SET) != 0) {
        fclose(input);
        return false;
    }
    u8* bytes = (u8*)malloc((size_t)end);
    if (bytes == NULL) {
        fclose(input);
        return false;
    }
    bool success = fread(bytes, 1, (size_t)end, input) == (size_t)end;
    if (fclose(input) != 0)
        success = false;
    if (!success) {
        free(bytes);
        return false;
    }
    *data = bytes;
    *size = (size_t)end;
    return true;
}

static bool load_container(DolMemoryCard* card) {
    u8* bytes = NULL;
    size_t size = 0;
    if (!read_whole_file(card->path, &bytes, &size))
        return false;

    bool valid = size >= DOL_CARD_HEADER_SIZE &&
                 memcmp(bytes, k_card_magic, sizeof k_card_magic) == 0 &&
                 read_be32(bytes + 8) == DOL_CARD_CONTAINER_VERSION &&
                 valid_size_mbits(read_be16(bytes + 12)) &&
                 read_be32(bytes + 16) == DOL_CARD_SECTOR_SIZE;
    if (!valid) {
        free(bytes);
        return false;
    }

    u16 size_mbits = read_be16(bytes + 12);
    u32 file_count = read_be32(bytes + 28);
    u32 body_size = read_be32(bytes + 32);
    valid = file_count <= DOL_CARD_MAX_FILES &&
            body_size == size - DOL_CARD_HEADER_SIZE &&
            read_be32(bytes + 36) ==
                hash_bytes(bytes + DOL_CARD_HEADER_SIZE, body_size);
    if (!valid) {
        free(bytes);
        return false;
    }

    card->size_mbits = size_mbits;
    card->encoding = read_be16(bytes + 14);
    card->serial = read_be64(bytes + 20);

    const u8* record = bytes + DOL_CARD_HEADER_SIZE;
    const u8* end = bytes + size;
    u32 used_blocks = 0;
    for (u32 entry = 0; entry < file_count; entry++) {
        if ((size_t)(end - record) < DOL_CARD_RECORD_SIZE) {
            valid = false;
            break;
        }
        u16 file_no = read_be16(record);
        u32 length = read_be32(record + 4);
        if (file_no >= DOL_CARD_MAX_FILES || card->files[file_no].used ||
            length > (size_t)(end - record - DOL_CARD_RECORD_SIZE)) {
            valid = false;
            break;
        }

        DolMemoryCardFile* file = &card->files[file_no];
        file->data = (u8*)malloc(length);
        if (file->data == NULL) {
            valid = false;
            break;
        }
        memcpy(file->data, record + DOL_CARD_RECORD_SIZE, length);
        if (read_be32(record + 64) != hash_bytes(file->data, length)) {
            valid = false;
            break;
        }

        file->used = true;
        file->stat.length = length;
        file->stat.time = read_be32(record + 8);
        memcpy(file->stat.file_name, record + 12, DOL_CARD_FILENAME_MAX);
        file->stat.file_name[DOL_CARD_FILENAME_MAX] = '\0';
        memcpy(file->stat.game_code, record + 44, 4);
        memcpy(file->stat.company, record + 48, 2);
        file->stat.banner_format = record[50];
        file->stat.permission = record[51];
        file->stat.icon_address = read_be32(record + 52);
        file->stat.icon_format = read_be16(record + 56);
        file->stat.icon_speed = read_be16(record + 58);
        file->stat.comment_address = read_be32(record + 60);

        used_blocks += file_blocks(length);
        record += DOL_CARD_RECORD_SIZE + length;
    }
    valid = valid && record == end &&
            used_blocks <= (u32)size_mbits * 16u - DOL_CARD_SYSTEM_BLOCKS;

    free(bytes);
    if (!valid)
        clear_files(card);
    return valid;
}

bool dol_card_validate(const char* path) {
    if (path == NULL || path[0] == '\0')
        return false;
    DolMemoryCard card = {0};
    card.path = (char*)path;
    bool valid = load_container(&card);
    clear_files(&card);
    return valid;
}

static u32 card_time(void) {
    const time_t gamecube_epoch = (time_t)946684800;
    time_t now = time(NULL);
    return now > gamecube_epoch ? (u32)(now - gamecube_epoch) : 0u;
}

DolMemoryCard* dol_card_open(const DolMemoryCardConfig* config) {
    if (config == NULL)
        return NULL;

    DolMemoryCard* card = (DolMemoryCard*)calloc(1, sizeof(*card));
    if (card == NULL)
        return NULL;
#ifndef _WIN32
    card->lock_fd = -1;
#endif
    card->path = duplicate_string(config->path);
    if (config->path != NULL && config->path[0] != '\0' &&
        card->path == NULL) {
        free(card);
        return NULL;
    }
    card->size_mbits = config->size_mbits != 0 ? config->size_mbits : 4u;
    card->encoding = config->encoding;
    card->serial = config->serial;
    memcpy(card->game_code, config->game_code, sizeof card->game_code);
    memcpy(card->company, config->company, sizeof card->company);
    card->last_result = DOL_CARD_RESULT_NO_CARD;

    if (!valid_size_mbits(card->size_mbits)) {
        dol_card_close(card);
        return NULL;
    }

    if (!lock_card(card)) {
        dol_card_close(card);
        return NULL;
    }
    errno = 0;
    FILE* existing = card->path != NULL ? fopen(card->path, "rb") : NULL;
    if (existing != NULL) {
        fclose(existing);
        if (!load_container(card)) {
            fprintf(stderr, "[card] invalid card container: %s\n", card->path);
            dol_card_close(card);
            return NULL;
        }
    } else {
        if (card->path != NULL && errno != ENOENT) {
            fprintf(stderr, "[card] cannot read %s; refusing to replace it\n", card->path);
            dol_card_close(card);
            return NULL;
        }
        if (card->serial == 0)
            card->serial = generated_serial(card->path);
        if (write_container(card) != CARD_WRITE_SYNCED) {
            fprintf(stderr, "[card] could not create card container: %s\n",
                    card->path != NULL ? card->path : "(memory)");
            dol_card_close(card);
            return NULL;
        }
    }
    return card;
}

void dol_card_close(DolMemoryCard* card) {
    if (card == NULL)
        return;
    clear_files(card);
#ifdef _WIN32
    if (card->lock_handle != NULL)
        CloseHandle(card->lock_handle);
#else
    if (card->lock_fd >= 0)
        close(card->lock_fd);
#endif
    // Do not unlink the lock file: another opener may already hold its inode.
    free(card->path);
    free(card);
}

DolCardResult dol_card_probe(DolMemoryCard* card, u16* size_mbits,
                             u32* sector_size) {
    if (card == NULL)
        return DOL_CARD_RESULT_NO_CARD;
    if (size_mbits != NULL)
        *size_mbits = card->size_mbits;
    if (sector_size != NULL)
        *sector_size = DOL_CARD_SECTOR_SIZE;
    return set_result(card, DOL_CARD_RESULT_READY, 0);
}

DolCardResult dol_card_mount(DolMemoryCard* card) {
    if (card == NULL)
        return DOL_CARD_RESULT_NO_CARD;
    card->mounted = true;
    return set_result(card, DOL_CARD_RESULT_READY,
                      DOL_CARD_SYSTEM_BLOCKS * DOL_CARD_SECTOR_SIZE);
}

DolCardResult dol_card_unmount(DolMemoryCard* card) {
    if (card == NULL)
        return DOL_CARD_RESULT_NO_CARD;
    card->mounted = false;
    return set_result(card, DOL_CARD_RESULT_READY, 0);
}

DolCardResult dol_card_check(DolMemoryCard* card) {
    if (!card_is_mounted(card))
        return set_result(card, DOL_CARD_RESULT_NO_CARD, 0);
    return set_result(card, DOL_CARD_RESULT_READY, 0);
}

DolCardResult dol_card_format(DolMemoryCard* card) {
    if (!card_is_mounted(card))
        return set_result(card, DOL_CARD_RESULT_NO_CARD, 0);

    DolMemoryCardFile previous[DOL_CARD_MAX_FILES];
    memcpy(previous, card->files, sizeof previous);
    memset(card->files, 0, sizeof card->files);
    CardWriteResult written = write_container(card);
    if (written == CARD_WRITE_FAILED) {
        memcpy(card->files, previous, sizeof card->files);
        return set_result(card, DOL_CARD_RESULT_IO_ERROR, 0);
    }
    for (u32 i = 0; i < DOL_CARD_MAX_FILES; i++)
        free(previous[i].data);
    if (written != CARD_WRITE_SYNCED)
        return set_result(card, DOL_CARD_RESULT_IO_ERROR, 0);
    return set_result(card, DOL_CARD_RESULT_READY,
                      DOL_CARD_SYSTEM_BLOCKS * DOL_CARD_SECTOR_SIZE);
}

DolCardResult dol_card_free_space(DolMemoryCard* card, u32* bytes_free,
                                  u32* files_free) {
    if (!card_is_mounted(card))
        return set_result(card, DOL_CARD_RESULT_NO_CARD, 0);

    u32 blocks_used = 0;
    u32 files_used = 0;
    for (u32 i = 0; i < DOL_CARD_MAX_FILES; i++) {
        if (!card->files[i].used)
            continue;
        blocks_used += file_blocks(card->files[i].stat.length);
        files_used++;
    }
    if (bytes_free != NULL)
        *bytes_free =
            (card_user_blocks(card) - blocks_used) * DOL_CARD_SECTOR_SIZE;
    if (files_free != NULL)
        *files_free = DOL_CARD_MAX_FILES - files_used;
    return set_result(card, DOL_CARD_RESULT_READY, 0);
}

DolCardResult dol_card_open_file(DolMemoryCard* card, const char* name,
                                 s32* file_no, u32* length) {
    if (!card_is_mounted(card))
        return set_result(card, DOL_CARD_RESULT_NO_CARD, 0);
    s32 index = find_file(card, name);
    if (index < 0)
        return set_result(card, DOL_CARD_RESULT_NO_FILE, 0);
    if (file_no != NULL)
        *file_no = index;
    if (length != NULL)
        *length = card->files[index].stat.length;
    return set_result(card, DOL_CARD_RESULT_READY, 0);
}

DolCardResult dol_card_create_file(DolMemoryCard* card, const char* name,
                                   u32 length, s32* file_no) {
    if (!card_is_mounted(card))
        return set_result(card, DOL_CARD_RESULT_NO_CARD, 0);
    if (name == NULL)
        return set_result(card, DOL_CARD_RESULT_FATAL, 0);
    if (strlen(name) > DOL_CARD_FILENAME_MAX)
        return set_result(card, DOL_CARD_RESULT_NAME_TOO_LONG, 0);
    if (length == 0 || length % DOL_CARD_SECTOR_SIZE != 0)
        return set_result(card, DOL_CARD_RESULT_FATAL, 0);
    if (find_file(card, name) >= 0)
        return set_result(card, DOL_CARD_RESULT_EXISTS, 0);

    u32 bytes_free = 0;
    u32 files_free = 0;
    dol_card_free_space(card, &bytes_free, &files_free);
    if (files_free == 0)
        return set_result(card, DOL_CARD_RESULT_NO_ENTRY, 0);
    if (length > bytes_free)
        return set_result(card, DOL_CARD_RESULT_INSUFFICIENT_SPACE, 0);

    s32 index = -1;
    for (u32 i = 0; i < DOL_CARD_MAX_FILES; i++) {
        if (!card->files[i].used) {
            index = (s32)i;
            break;
        }
    }
    if (index < 0)
        return set_result(card, DOL_CARD_RESULT_NO_ENTRY, 0);

    DolMemoryCardFile* file = &card->files[index];
    file->data = (u8*)malloc(length);
    if (file->data == NULL)
        return set_result(card, DOL_CARD_RESULT_IO_ERROR, 0);
    memset(file->data, 0xFF, length);
    file->used = true;
    memcpy(file->stat.game_code, card->game_code, sizeof card->game_code);
    memcpy(file->stat.company, card->company, sizeof card->company);
    memcpy(file->stat.file_name, name, strlen(name) + 1u);
    file->stat.length = length;
    file->stat.time = card_time();
    file->stat.permission = 4u;
    file->stat.icon_address = 0xFFFFFFFFu;
    file->stat.icon_speed = 1u;
    file->stat.comment_address = 0xFFFFFFFFu;

    CardWriteResult written = write_container(card);
    if (written != CARD_WRITE_SYNCED) {
        if (written == CARD_WRITE_FAILED) {
            free(file->data);
            memset(file, 0, sizeof(*file));
        }
        return set_result(card, DOL_CARD_RESULT_IO_ERROR, 0);
    }
    if (file_no != NULL)
        *file_no = index;
    return set_result(card, DOL_CARD_RESULT_READY,
                      2u * DOL_CARD_SECTOR_SIZE);
}

DolCardResult dol_card_close_file(DolMemoryCard* card, s32 file_no) {
    if (!card_is_mounted(card))
        return set_result(card, DOL_CARD_RESULT_NO_CARD, 0);
    if (file_no < 0 || file_no >= (s32)DOL_CARD_MAX_FILES ||
        !card->files[file_no].used ||
        !file_matches_game(card, &card->files[file_no]))
        return set_result(card, DOL_CARD_RESULT_FATAL, 0);
    return set_result(card, DOL_CARD_RESULT_READY, 0);
}

DolCardResult dol_card_delete_file(DolMemoryCard* card, const char* name) {
    if (!card_is_mounted(card))
        return set_result(card, DOL_CARD_RESULT_NO_CARD, 0);
    s32 index = find_file(card, name);
    if (index < 0)
        return set_result(card, DOL_CARD_RESULT_NO_FILE, 0);

    DolMemoryCardFile removed = card->files[index];
    memset(&card->files[index], 0, sizeof(card->files[index]));
    CardWriteResult written = write_container(card);
    if (written == CARD_WRITE_FAILED) {
        card->files[index] = removed;
        return set_result(card, DOL_CARD_RESULT_IO_ERROR, 0);
    }
    free(removed.data);
    if (written != CARD_WRITE_SYNCED)
        return set_result(card, DOL_CARD_RESULT_IO_ERROR, 0);
    return set_result(card, DOL_CARD_RESULT_READY,
                      2u * DOL_CARD_SECTOR_SIZE);
}

static DolCardResult validate_file_range(DolMemoryCard* card, s32 file_no,
                                         u32 offset, u32 length) {
    if (!card_is_mounted(card))
        return DOL_CARD_RESULT_NO_CARD;
    if (file_no < 0 || file_no >= (s32)DOL_CARD_MAX_FILES ||
        !card->files[file_no].used ||
        !file_matches_game(card, &card->files[file_no]))
        return DOL_CARD_RESULT_NO_FILE;
    u32 file_length = card->files[file_no].stat.length;
    if (offset > file_length || length > file_length - offset)
        return DOL_CARD_RESULT_LIMIT;
    return DOL_CARD_RESULT_READY;
}

DolCardResult dol_card_read_file(DolMemoryCard* card, s32 file_no, u32 offset,
                                 void* data, u32 length) {
    DolCardResult result =
        validate_file_range(card, file_no, offset, length);
    if (result != DOL_CARD_RESULT_READY)
        return set_result(card, result, 0);
    if (data == NULL && length != 0)
        return set_result(card, DOL_CARD_RESULT_FATAL, 0);
    if (length != 0)
        memcpy(data, card->files[file_no].data + offset, length);
    return set_result(card, DOL_CARD_RESULT_READY, length);
}

DolCardResult dol_card_write_file(DolMemoryCard* card, s32 file_no, u32 offset,
                                  const void* data, u32 length) {
    DolCardResult result =
        validate_file_range(card, file_no, offset, length);
    if (result != DOL_CARD_RESULT_READY)
        return set_result(card, result, 0);
    if (data == NULL && length != 0)
        return set_result(card, DOL_CARD_RESULT_FATAL, 0);

    u8* destination = card->files[file_no].data + offset;
    u8* old_data = NULL;
    if (length != 0) {
        old_data = (u8*)malloc(length);
        if (old_data == NULL)
            return set_result(card, DOL_CARD_RESULT_IO_ERROR, 0);
        memcpy(old_data, destination, length);
        memcpy(destination, data, length);
    }
    u32 previous_time = card->files[file_no].stat.time;
    card->files[file_no].stat.time = card_time();
    CardWriteResult written = write_container(card);
    if (written != CARD_WRITE_SYNCED) {
        if (written == CARD_WRITE_FAILED && length != 0)
            memcpy(destination, old_data, length);
        if (written == CARD_WRITE_FAILED)
            card->files[file_no].stat.time = previous_time;
        free(old_data);
        return set_result(card, DOL_CARD_RESULT_IO_ERROR, 0);
    }
    free(old_data);
    return set_result(card, DOL_CARD_RESULT_READY, length);
}

DolCardResult dol_card_get_status(DolMemoryCard* card, s32 file_no,
                                  DolMemoryCardStat* stat) {
    DolCardResult result = validate_file_range(card, file_no, 0, 0);
    if (result != DOL_CARD_RESULT_READY)
        return set_result(card, result, 0);
    if (stat == NULL)
        return set_result(card, DOL_CARD_RESULT_FATAL, 0);
    *stat = card->files[file_no].stat;
    return set_result(card, DOL_CARD_RESULT_READY, 0);
}

DolCardResult dol_card_set_status(DolMemoryCard* card, s32 file_no,
                                  const DolMemoryCardStat* stat) {
    DolCardResult result = validate_file_range(card, file_no, 0, 0);
    if (result != DOL_CARD_RESULT_READY)
        return set_result(card, result, 0);
    if (stat == NULL)
        return set_result(card, DOL_CARD_RESULT_FATAL, 0);

    DolMemoryCardStat previous = card->files[file_no].stat;
    card->files[file_no].stat.banner_format = stat->banner_format;
    card->files[file_no].stat.icon_address = stat->icon_address;
    card->files[file_no].stat.icon_format = stat->icon_format;
    card->files[file_no].stat.icon_speed = stat->icon_speed;
    card->files[file_no].stat.comment_address = stat->comment_address;
    card->files[file_no].stat.time = card_time();
    CardWriteResult written = write_container(card);
    if (written != CARD_WRITE_SYNCED) {
        if (written == CARD_WRITE_FAILED)
            card->files[file_no].stat = previous;
        return set_result(card, DOL_CARD_RESULT_IO_ERROR, 0);
    }
    return set_result(card, DOL_CARD_RESULT_READY, DOL_CARD_SECTOR_SIZE);
}

u64 dol_card_serial(const DolMemoryCard* card) {
    return card != NULL ? card->serial : 0;
}

s32 dol_card_last_result(const DolMemoryCard* card) {
    return card != NULL ? card->last_result : DOL_CARD_RESULT_NO_CARD;
}

u32 dol_card_transferred_bytes(const DolMemoryCard* card) {
    return card != NULL ? card->transferred_bytes : 0;
}
